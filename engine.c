/*
 * engine.c — High-performance bitboard engine for Hex Connect-6 triangle proof
 *
 * Compile: cc -O3 -march=native -shared -fPIC -o engine.so engine.c
 *
 * Exports functions via C ABI for use with Python ctypes.
 * Features:
 *   - Bitboard win detection (5 shifts + 5 ANDs per axis)
 *   - Pre-allocated candidate tracking (no malloc in hot path)
 *   - Undo stack for backtracking search
 *   - Threat detection (count 4+ and 5+ in a row)
 *   - Heuristic move scoring
 *   - Batch random rollout (entirely in C)
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* ===================================================================
 * Constants
 * =================================================================== */

#define SIZE       31
#define OFF        15
#define WIN_LEN    6
#define MAX_STONES 200
#define MAX_CANDS  1200  /* radius 3 generates more candidates */
#define MAX_UNDO   250
#define CAND_RADIUS 3  /* radius 3 allows 1-hex-gap placements for preemptives */

/* Neighbor offsets for candidate expansion (pre-computed) */
typedef struct { int dq, dr; } Offset;

static Offset NEIGHBOR_OFFSETS[50];
static int    NUM_NEIGHBORS = 0;

static int hex_distance(int dq, int dr) {
    int a = abs(dq), b = abs(dr), c = abs(dq + dr);
    if (a >= b && a >= c) return a;
    if (b >= c) return b;
    return c;
}

__attribute__((constructor))
static void init_offsets(void) {
    NUM_NEIGHBORS = 0;
    for (int dq = -CAND_RADIUS; dq <= CAND_RADIUS; dq++) {
        for (int dr = -CAND_RADIUS; dr <= CAND_RADIUS; dr++) {
            int d = hex_distance(dq, dr);
            if (d >= 1 && d <= CAND_RADIUS) {
                NEIGHBOR_OFFSETS[NUM_NEIGHBORS].dq = dq;
                NEIGHBOR_OFFSETS[NUM_NEIGHBORS].dr = dr;
                NUM_NEIGHBORS++;
            }
        }
    }
}

/* ===================================================================
 * Zobrist tables (pre-allocated, initialized once)
 * =================================================================== */

static uint64_t ZOBRIST[2][SIZE][SIZE];  /* [player][qi][ri] */
static int zobrist_initialized = 0;

static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

static void init_zobrist(void) {
    if (zobrist_initialized) return;
    uint64_t state = 0x12345678ABCDEF01ULL;
    for (int p = 0; p < 2; p++)
        for (int qi = 0; qi < SIZE; qi++)
            for (int ri = 0; ri < SIZE; ri++)
                ZOBRIST[p][qi][ri] = xorshift64(&state);
    zobrist_initialized = 1;
}

/* ===================================================================
 * Board state
 * =================================================================== */

typedef struct {
    int qi, ri, player;
    int cands_added[50];  /* indices into flat array: qi*SIZE+ri */
    int cands_added_count;
    int cand_was_present;
    int stones_this_turn, stones_per_turn, turn, current_player;
    int winner;
    uint64_t zhash;
} UndoRecord;

typedef struct {
    /* Bitboards: one per axis-row per player */
    uint64_t bits_h[SIZE][2];    /* axis (1,0): row=ri, bit=qi */
    uint64_t bits_v[SIZE][2];    /* axis (0,1): row=qi, bit=ri */
    uint64_t bits_d[SIZE * 2][2]; /* axis (1,-1): row=qi+ri, bit=qi */

    uint8_t board[SIZE][SIZE];   /* 0=empty, 1=X, 2=O */

    /* Candidate tracking */
    uint8_t is_cand[SIZE][SIZE]; /* boolean: is this cell a candidate? */
    int16_t cand_qi[MAX_CANDS];
    int16_t cand_ri[MAX_CANDS];
    int     cand_count;

    /* Turn state */
    int current_player;
    int stones_this_turn;
    int stones_per_turn;
    int turn;
    int total_stones;
    int winner;  /* -1=none, 0=X, 1=O */
    int max_stones;

    /* Zobrist */
    uint64_t zhash;

    /* Undo stack */
    UndoRecord undo_stack[MAX_UNDO];
    int undo_top;
} Board;

/* ===================================================================
 * Board operations
 * =================================================================== */

static void cand_add(Board *b, int qi, int ri) {
    if (qi < 0 || qi >= SIZE || ri < 0 || ri >= SIZE) return;
    if (b->board[qi][ri] != 0) return;  /* occupied */
    if (b->is_cand[qi][ri]) return;     /* already candidate */
    b->is_cand[qi][ri] = 1;
    b->cand_qi[b->cand_count] = (int16_t)qi;
    b->cand_ri[b->cand_count] = (int16_t)ri;
    b->cand_count++;
}

static void cand_remove_at(Board *b, int idx) {
    b->cand_count--;
    b->is_cand[b->cand_qi[idx]][b->cand_ri[idx]] = 0;
    /* Swap with last */
    b->cand_qi[idx] = b->cand_qi[b->cand_count];
    b->cand_ri[idx] = b->cand_ri[b->cand_count];
}

static int cand_find(Board *b, int qi, int ri) {
    for (int i = 0; i < b->cand_count; i++) {
        if (b->cand_qi[i] == qi && b->cand_ri[i] == ri)
            return i;
    }
    return -1;
}

/* Inline win check */
static inline int check_win_axis(uint64_t bits) {
    return (bits & (bits >> 1) & (bits >> 2) & (bits >> 3) & (bits >> 4) & (bits >> 5)) != 0;
}

void board_reset(Board *b) {
    init_zobrist();
    memset(b->bits_h, 0, sizeof(b->bits_h));
    memset(b->bits_v, 0, sizeof(b->bits_v));
    memset(b->bits_d, 0, sizeof(b->bits_d));
    memset(b->board, 0, sizeof(b->board));
    memset(b->is_cand, 0, sizeof(b->is_cand));
    b->cand_count = 0;
    b->current_player = 0;
    b->stones_this_turn = 0;
    b->stones_per_turn = 1;
    b->turn = 0;
    b->total_stones = 0;
    b->winner = -1;
    b->max_stones = MAX_STONES;
    b->zhash = 0;
    b->undo_top = 0;
}

void board_place(Board *b, int q, int r) {
    int qi = q + OFF, ri = r + OFF;
    int p = b->current_player;
    int pv = p + 1;

    /* Push undo record */
    UndoRecord *rec = &b->undo_stack[b->undo_top++];
    rec->qi = qi;
    rec->ri = ri;
    rec->player = p;
    rec->cands_added_count = 0;
    rec->cand_was_present = b->is_cand[qi][ri];
    rec->stones_this_turn = b->stones_this_turn;
    rec->stones_per_turn = b->stones_per_turn;
    rec->turn = b->turn;
    rec->current_player = b->current_player;
    rec->winner = b->winner;
    rec->zhash = b->zhash;

    /* Place stone */
    b->board[qi][ri] = (uint8_t)pv;
    b->bits_h[ri][p] |= (1ULL << qi);
    b->bits_v[qi][p] |= (1ULL << ri);
    b->bits_d[qi + ri][p] |= (1ULL << qi);

    /* Remove from candidates if present */
    if (b->is_cand[qi][ri]) {
        int idx = cand_find(b, qi, ri);
        if (idx >= 0) cand_remove_at(b, idx);
        else b->is_cand[qi][ri] = 0;  /* safety */
    }

    /* Expand candidates */
    for (int n = 0; n < NUM_NEIGHBORS; n++) {
        int nqi = qi + NEIGHBOR_OFFSETS[n].dq;
        int nri = ri + NEIGHBOR_OFFSETS[n].dr;
        if (nqi >= 0 && nqi < SIZE && nri >= 0 && nri < SIZE) {
            if (b->board[nqi][nri] == 0 && !b->is_cand[nqi][nri]) {
                cand_add(b, nqi, nri);
                rec->cands_added[rec->cands_added_count++] = nqi * SIZE + nri;
            }
        }
    }

    /* Win check */
    int win = 0;
    if (check_win_axis(b->bits_h[ri][p])) win = 1;
    else if (check_win_axis(b->bits_v[qi][p])) win = 1;
    else if (check_win_axis(b->bits_d[qi + ri][p])) win = 1;

    /* Zobrist */
    b->zhash ^= ZOBRIST[p][qi][ri];

    b->total_stones++;
    if (win) b->winner = p;

    /* Advance turn */
    b->stones_this_turn++;
    if (b->stones_this_turn >= b->stones_per_turn) {
        b->turn++;
        b->current_player = b->turn & 1;
        b->stones_this_turn = 0;
        b->stones_per_turn = 2;
    }
}

void board_undo(Board *b) {
    if (b->undo_top <= 0) return;
    UndoRecord *rec = &b->undo_stack[--b->undo_top];

    int qi = rec->qi, ri = rec->ri, p = rec->player;

    b->board[qi][ri] = 0;
    b->bits_h[ri][p] &= ~(1ULL << qi);
    b->bits_v[qi][p] &= ~(1ULL << ri);
    b->bits_d[qi + ri][p] &= ~(1ULL << qi);

    /* Remove added candidates */
    for (int i = rec->cands_added_count - 1; i >= 0; i--) {
        int flat = rec->cands_added[i];
        int nqi = flat / SIZE, nri = flat % SIZE;
        if (b->is_cand[nqi][nri]) {
            int idx = cand_find(b, nqi, nri);
            if (idx >= 0) cand_remove_at(b, idx);
        }
    }

    /* Restore candidate if was present */
    if (rec->cand_was_present) {
        cand_add(b, qi, ri);
    }

    b->stones_this_turn = rec->stones_this_turn;
    b->stones_per_turn = rec->stones_per_turn;
    b->turn = rec->turn;
    b->current_player = rec->current_player;
    b->winner = rec->winner;
    b->zhash = rec->zhash;
    b->total_stones--;
}

/* ===================================================================
 * Setup
 * =================================================================== */

static void place_raw(Board *b, int q, int r, int player) {
    /* Raw placement without turn advancement — for setup only */
    int qi = q + OFF, ri = r + OFF;
    int pv = player + 1;
    b->board[qi][ri] = (uint8_t)pv;
    b->bits_h[ri][player] |= (1ULL << qi);
    b->bits_v[qi][player] |= (1ULL << ri);
    b->bits_d[qi + ri][player] |= (1ULL << qi);
    b->zhash ^= ZOBRIST[player][qi][ri];

    if (b->is_cand[qi][ri]) {
        int idx = cand_find(b, qi, ri);
        if (idx >= 0) cand_remove_at(b, idx);
    }

    for (int n = 0; n < NUM_NEIGHBORS; n++) {
        int nqi = qi + NEIGHBOR_OFFSETS[n].dq;
        int nri = ri + NEIGHBOR_OFFSETS[n].dr;
        if (nqi >= 0 && nqi < SIZE && nri >= 0 && nri < SIZE) {
            if (b->board[nqi][nri] == 0 && !b->is_cand[nqi][nri]) {
                cand_add(b, nqi, nri);
            }
        }
    }
    b->total_stones++;
}

void board_setup_triangle(Board *b) {
    board_reset(b);
    place_raw(b, 0, 0, 0);   /* X at center */
    place_raw(b, 1, 0, 0);   /* X at A0 */
    place_raw(b, 0, 1, 0);   /* X at A1 */
    b->current_player = 1;   /* O to move */
    b->stones_this_turn = 0;
    b->stones_per_turn = 2;
    b->turn = 1;
    b->total_stones = 3;
}

/* ===================================================================
 * Threat detection
 * =================================================================== */

static int count_line_dir(Board *b, int qi, int ri, int dq, int dr, int player) {
    int pv = player + 1;
    int c = 0;
    int nqi = qi + dq, nri = ri + dr;
    while (nqi >= 0 && nqi < SIZE && nri >= 0 && nri < SIZE && b->board[nqi][nri] == pv) {
        c++;
        nqi += dq;
        nri += dr;
    }
    return c;
}

int board_max_line_through(Board *b, int qi, int ri, int player) {
    /* Max consecutive including (qi,ri) on any axis */
    static const int axes[3][2] = {{1,0},{0,1},{1,-1}};
    int best = 0;
    for (int a = 0; a < 3; a++) {
        int dq = axes[a][0], dr = axes[a][1];
        int c = 1 + count_line_dir(b, qi, ri, dq, dr, player)
                  + count_line_dir(b, qi, ri, -dq, -dr, player);
        if (c > best) best = c;
    }
    return best;
}

int board_would_win(Board *b, int qi, int ri, int player) {
    /* Check if placing player at (qi,ri) makes 6+ (without actually placing) */
    static const int axes[3][2] = {{1,0},{0,1},{1,-1}};
    int pv = player + 1;
    for (int a = 0; a < 3; a++) {
        int dq = axes[a][0], dr = axes[a][1];
        int c = 1;
        int nqi = qi + dq, nri = ri + dr;
        while (nqi >= 0 && nqi < SIZE && nri >= 0 && nri < SIZE && b->board[nqi][nri] == pv) {
            c++; if (c >= 6) return 1;
            nqi += dq; nri += dr;
        }
        nqi = qi - dq; nri = ri - dr;
        while (nqi >= 0 && nqi < SIZE && nri >= 0 && nri < SIZE && b->board[nqi][nri] == pv) {
            c++; if (c >= 6) return 1;
            nqi -= dq; nri -= dr;
        }
    }
    return 0;
}

int board_has_winning_move(Board *b, int player) {
    /* Check if player can win in one stone */
    for (int i = 0; i < b->cand_count; i++) {
        int qi = b->cand_qi[i], ri = b->cand_ri[i];
        if (board_would_win(b, qi, ri, player))
            return 1;
    }
    return 0;
}

int board_count_winning_moves(Board *b, int player) {
    int count = 0;
    for (int i = 0; i < b->cand_count; i++) {
        int qi = b->cand_qi[i], ri = b->cand_ri[i];
        if (board_would_win(b, qi, ri, player))
            count++;
    }
    return count;
}

int board_is_forcing(Board *b, int qi, int ri, int player) {
    /* Does placing here create a 4+ for player or block opponent's 4+? */
    static const int axes[3][2] = {{1,0},{0,1},{1,-1}};
    int pv = player + 1;
    int ov = 2 - player;

    for (int a = 0; a < 3; a++) {
        int dq = axes[a][0], dr = axes[a][1];
        /* Own line length */
        int c = 1 + count_line_dir(b, qi, ri, dq, dr, player)
                  + count_line_dir(b, qi, ri, -dq, -dr, player);
        if (c >= 4) return 1;  /* Creates threat */
        /* Opponent line length (blocking) */
        c = 1 + count_line_dir(b, qi, ri, dq, dr, 1 - player)
              + count_line_dir(b, qi, ri, -dq, -dr, 1 - player);
        if (c >= 4) return 1;  /* Blocks threat */
    }
    return 0;
}

/* ===================================================================
 * Move scoring (heuristic)
 * =================================================================== */

/* Count how many open lines of length N pass through a cell.
 * "Open" means the line can still extend to 6 (not blocked on both ends). */
static int count_open_lines(Board *b, int qi, int ri, int player, int min_len) {
    static const int axes[3][2] = {{1,0},{0,1},{1,-1}};
    int pv = player + 1;
    int count = 0;
    for (int a = 0; a < 3; a++) {
        int dq = axes[a][0], dr = axes[a][1];
        int c = 1 + count_line_dir(b, qi, ri, dq, dr, player)
                  + count_line_dir(b, qi, ri, -dq, -dr, player);
        if (c >= min_len) count++;
    }
    return count;
}

/* Check if a cell is adjacent to any stone of either player */
static int has_stone_neighbor(Board *b, int qi, int ri) {
    static const int dirs[6][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,-1},{-1,1}};
    for (int d = 0; d < 6; d++) {
        int nqi = qi + dirs[d][0], nri = ri + dirs[d][1];
        if (nqi >= 0 && nqi < SIZE && nri >= 0 && nri < SIZE && b->board[nqi][nri] != 0)
            return 1;
    }
    return 0;
}

int board_move_score(Board *b, int qi, int ri) {
    static const int axes[3][2] = {{1,0},{0,1},{1,-1}};
    int p = b->current_player;
    int pv = p + 1;
    int score = 0;
    int best_my = 0, best_opp = 0;
    int my_lines_4 = 0, opp_lines_4 = 0;
    int my_lines_3 = 0, opp_lines_3 = 0;

    for (int a = 0; a < 3; a++) {
        int dq = axes[a][0], dr = axes[a][1];
        int c = 1 + count_line_dir(b, qi, ri, dq, dr, p)
                  + count_line_dir(b, qi, ri, -dq, -dr, p);
        if (c > best_my) best_my = c;
        if (c >= 4) my_lines_4++;
        if (c >= 3) my_lines_3++;

        c = 1 + count_line_dir(b, qi, ri, dq, dr, 1 - p)
              + count_line_dir(b, qi, ri, -dq, -dr, 1 - p);
        if (c > best_opp) best_opp = c;
        if (c >= 4) opp_lines_4++;
        if (c >= 3) opp_lines_3++;
    }

    /* Immediate win */
    if (best_my >= 6) return 200000;

    /* Must-block opponent win */
    if (best_opp >= 6) return 150000;

    /* Fork creation: 2+ lines of 4 = guaranteed win next turn */
    if (my_lines_4 >= 2) score += 80000;
    else if (my_lines_4 >= 1) score += 8000;

    /* Near-win: 5 in a row (need one more) */
    if (best_my >= 5) score += 20000;

    /* Block opponent's near-win */
    if (best_opp >= 5) score += 15000;

    /* Fork defense: block opponent creating 2+ lines of 4 */
    if (opp_lines_4 >= 2) score += 40000;
    else if (opp_lines_4 >= 1) score += 3000;

    /* Multi-threat: creating multiple 3-lines (structural advantage) */
    if (my_lines_3 >= 2) score += 2000;
    else if (my_lines_3 >= 1) score += 300;

    /* Block opponent 3-lines */
    if (opp_lines_3 >= 2) score += 1000;

    /* Base line value */
    score += best_my * 15 + best_opp * 8;

    /* Proximity: moves near existing stones are much better */
    if (!has_stone_neighbor(b, qi, ri)) score -= 200;

    /* Center proximity (mild) */
    int dq = qi - OFF, dr = ri - OFF;
    score -= (dq > 0 ? dq : -dq) + (dr > 0 ? dr : -dr);

    return score;
}

/* Heuristic evaluation of a non-terminal position */
static float board_evaluate(Board *b) {
    static const int axes[3][2] = {{1,0},{0,1},{1,-1}};
    float score = 0.0f;

    /* Count threats and strong lines for each player */
    int x_threats5 = 0, o_threats5 = 0;  /* cells making 5+ in a row */
    int x_threats4 = 0, o_threats4 = 0;  /* cells making 4+ in a row */
    int x_lines3 = 0, o_lines3 = 0;     /* existing 3+ lines */

    /* Check all candidate cells for threats */
    for (int i = 0; i < b->cand_count; i++) {
        int qi = b->cand_qi[i], ri = b->cand_ri[i];
        for (int a = 0; a < 3; a++) {
            int dq = axes[a][0], dr = axes[a][1];
            /* X line through this empty cell */
            int cx = 1 + count_line_dir(b, qi, ri, dq, dr, 0)
                       + count_line_dir(b, qi, ri, -dq, -dr, 0);
            if (cx >= 6) x_threats5++;
            else if (cx >= 5) x_threats5++;
            else if (cx >= 4) x_threats4++;

            /* O line through this empty cell */
            int co = 1 + count_line_dir(b, qi, ri, dq, dr, 1)
                       + count_line_dir(b, qi, ri, -dq, -dr, 1);
            if (co >= 6) o_threats5++;
            else if (co >= 5) o_threats5++;
            else if (co >= 4) o_threats4++;
        }
    }

    /* Score: threats are very valuable */
    score += (x_threats5 - o_threats5) * 0.4f;
    score += (x_threats4 - o_threats4) * 0.1f;

    /* Clamp to (-0.95, 0.95) — never claim certainty from heuristic */
    if (score > 0.95f) score = 0.95f;
    if (score < -0.95f) score = -0.95f;

    return score;
}

/* ===================================================================
 * Get ordered moves (for Python to call)
 * =================================================================== */

int board_get_candidates(Board *b, int *out_q, int *out_r) {
    for (int i = 0; i < b->cand_count; i++) {
        out_q[i] = b->cand_qi[i] - OFF;
        out_r[i] = b->cand_ri[i] - OFF;
    }
    return b->cand_count;
}

int board_get_forcing_moves(Board *b, int *out_q, int *out_r) {
    int count = 0;
    int p = b->current_player;
    for (int i = 0; i < b->cand_count; i++) {
        int qi = b->cand_qi[i], ri = b->cand_ri[i];
        if (board_is_forcing(b, qi, ri, p)) {
            out_q[count] = qi - OFF;
            out_r[count] = ri - OFF;
            count++;
        }
    }
    return count;
}

int board_get_scored_moves(Board *b, int *out_q, int *out_r, int *out_score, int limit) {
    /* Return top-N moves sorted by score */
    int n = b->cand_count;
    if (n > MAX_CANDS) n = MAX_CANDS;

    /* Score all candidates */
    int scores[MAX_CANDS];
    for (int i = 0; i < n; i++) {
        scores[i] = board_move_score(b, b->cand_qi[i], b->cand_ri[i]);
    }

    /* Simple selection sort for top-N (N is small) */
    int count = 0;
    uint8_t used[MAX_CANDS];
    memset(used, 0, n);

    for (int k = 0; k < limit && k < n; k++) {
        int best_idx = -1, best_score = -999999;
        for (int i = 0; i < n; i++) {
            if (!used[i] && scores[i] > best_score) {
                best_score = scores[i];
                best_idx = i;
            }
        }
        if (best_idx < 0) break;
        used[best_idx] = 1;
        out_q[count] = b->cand_qi[best_idx] - OFF;
        out_r[count] = b->cand_ri[best_idx] - OFF;
        if (out_score) out_score[count] = best_score;
        count++;
    }
    return count;
}

/* ===================================================================
 * Alpha-beta solver (entirely in C for maximum speed)
 * =================================================================== */

static long long ab_nodes = 0;
static long long tt_hits = 0;

/* ===================================================================
 * Transposition table (replacement: always-replace by depth)
 * =================================================================== */

typedef struct {
    uint64_t key;     /* full Zobrist key for collision detection */
    int      depth;
    float    value;
    int      ste;
    int      flag;    /* 0=exact, 1=lower bound, 2=upper bound */
} TTEntry;

#define TT_SIZE (1 << 23)  /* 8M entries (~384 MB) */
#define TT_MASK (TT_SIZE - 1)
static TTEntry tt_table[TT_SIZE];

static void tt_clear(void) {
    memset(tt_table, 0, sizeof(tt_table));
}

static TTEntry *tt_probe(uint64_t key) {
    TTEntry *e = &tt_table[key & TT_MASK];
    if (e->key == key && e->depth > 0) return e;
    return NULL;
}

static void tt_store(uint64_t key, int depth, float value, int ste, int flag) {
    TTEntry *e = &tt_table[key & TT_MASK];
    /* Replace if deeper or same position */
    if (depth >= e->depth || e->key == key) {
        e->key = key;
        e->depth = depth;
        e->value = value;
        e->ste = ste;
        e->flag = flag;
    }
}

/* ===================================================================
 * Killer move table
 * =================================================================== */

#define MAX_PLY 200
static int killer_q[MAX_PLY][2];
static int killer_r[MAX_PLY][2];

static void killers_clear(void) {
    memset(killer_q, 0, sizeof(killer_q));
    memset(killer_r, 0, sizeof(killer_r));
}

static void record_killer(int ply, int q, int r) {
    if (ply >= MAX_PLY) return;
    if (killer_q[ply][0] == q && killer_r[ply][0] == r) return;
    killer_q[ply][1] = killer_q[ply][0];
    killer_r[ply][1] = killer_r[ply][0];
    killer_q[ply][0] = q;
    killer_r[ply][0] = r;
}

/* ===================================================================
 * Alpha-beta solver with TT + killers + LMR
 * =================================================================== */

typedef struct { float value; int ste; } ABResult;

static ABResult ab_solve(Board *b, int depth, float alpha, float beta, int ply) {
    ABResult result;
    ab_nodes++;

    /* Terminal */
    if (b->winner >= 0) {
        result.value = (b->winner == 0) ? 1.0f : -1.0f;
        result.ste = 0;
        return result;
    }
    if (depth <= 0) {
        /* Heuristic eval guides pruning; capped at ±0.9 so proven wins (±1.0) are distinguishable */
        result.value = board_evaluate(b);
        result.ste = 0;
        return result;
    }

    int p = b->current_player;
    int maximizing = (p == 0);

    /* TT probe */
    uint64_t key = b->zhash;
    TTEntry *tte = tt_probe(key);
    if (tte && tte->depth >= depth) {
        tt_hits++;
        if (tte->flag == 0) {  /* exact */
            result.value = tte->value;
            result.ste = tte->ste;
            return result;
        }
        if (tte->flag == 1 && tte->value >= beta) {  /* lower bound */
            result.value = tte->value;
            result.ste = tte->ste;
            return result;
        }
        if (tte->flag == 2 && tte->value <= alpha) {  /* upper bound */
            result.value = tte->value;
            result.ste = tte->ste;
            return result;
        }
    }

    /* Immediate win check */
    if (board_has_winning_move(b, p)) {
        result.value = maximizing ? 1.0f : -1.0f;
        result.ste = 1;
        tt_store(key, depth, result.value, result.ste, 0);
        return result;
    }

    /* Double threat at turn start: ≥3 winning cells = opponent can't block all */
    if (b->stones_this_turn == 0) {
        int wm = board_count_winning_moves(b, p);
        if (wm >= 3) {
            result.value = maximizing ? 1.0f : -1.0f;
            result.ste = 2;
            tt_store(key, depth, result.value, result.ste, 0);
            return result;
        }
    }

    /* Get moves: aggressive pruning for deeper effective search.
     * Human proofs only consider 5-8 moves per position. */
    int move_q[40], move_r[40], move_sc[40];
    int move_limit = (depth <= 3) ? 12 : (depth <= 6 ? 15 : 20);
    int nmoves = board_get_scored_moves(b, move_q, move_r, move_sc, move_limit);

    if (nmoves == 0) {
        result.value = 0.0f;
        result.ste = 0;
        return result;
    }

    /* Insert killer moves at front if legal */
    if (ply < MAX_PLY) {
        for (int k = 1; k >= 0; k--) {
            int kq = killer_q[ply][k], kr = killer_r[ply][k];
            if (kq == 0 && kr == 0) continue;
            /* Check if this killer is in the candidate set */
            int kqi = kq + OFF, kri = kr + OFF;
            if (kqi >= 0 && kqi < SIZE && kri >= 0 && kri < SIZE &&
                b->board[kqi][kri] == 0 && b->is_cand[kqi][kri]) {
                /* Move to front */
                for (int i = 0; i < nmoves; i++) {
                    if (move_q[i] == kq && move_r[i] == kr) {
                        /* Swap with position 0 (or 1 for second killer) */
                        int target = (k == 0) ? 0 : (nmoves > 1 ? 1 : 0);
                        int tq = move_q[target], tr = move_r[target], ts = move_sc[target];
                        move_q[target] = kq; move_r[target] = kr; move_sc[target] = move_sc[i];
                        move_q[i] = tq; move_r[i] = tr; move_sc[i] = ts;
                        break;
                    }
                }
            }
        }
    }

    float orig_alpha = alpha;
    float best_val = maximizing ? -2.0f : 2.0f;
    int best_ste = 0;

    for (int i = 0; i < nmoves; i++) {
        board_place(b, move_q[i], move_r[i]);

        ABResult child;
        /* Late move reduction */
        int do_full = 1;
        if (i >= 10 && depth >= 4) {
            child = ab_solve(b, depth - 3, alpha, beta, ply + 1);
            if (maximizing && child.value <= alpha) do_full = 0;
            else if (!maximizing && child.value >= beta) do_full = 0;
        }
        if (do_full) {
            child = ab_solve(b, depth - 1, alpha, beta, ply + 1);
        }

        board_undo(b);

        int ste = child.ste + 1;
        if (maximizing) {
            if (child.value > best_val || (child.value == best_val && ste < best_ste)) {
                best_val = child.value;
                best_ste = ste;
            }
            if (best_val > alpha) alpha = best_val;
        } else {
            if (child.value < best_val || (child.value == best_val && ste > best_ste)) {
                best_val = child.value;
                best_ste = ste;
            }
            if (best_val < beta) beta = best_val;
        }
        if (alpha >= beta) {
            record_killer(ply, move_q[i], move_r[i]);
            break;
        }
    }

    /* TT store */
    int flag;
    if (best_val <= orig_alpha) flag = 2;      /* upper bound */
    else if (best_val >= beta)  flag = 1;      /* lower bound */
    else                        flag = 0;      /* exact */
    tt_store(key, depth, best_val, best_ste, flag);

    result.value = best_val;
    result.ste = best_ste;
    return result;
}

/* Public API: single-depth solve */
long long c_ab_solve(Board *b, int depth, float *out_value, int *out_ste) {
    ab_nodes = 0;
    tt_hits = 0;
    tt_clear();
    killers_clear();
    ABResult r = ab_solve(b, depth, -2.0f, 2.0f, 0);
    *out_value = r.value;
    *out_ste = r.ste;
    return ab_nodes;
}

/* Public API: iterative deepening (keeps TT across depths) */
typedef struct {
    long long nodes;
    long long tt_hit;
    float     value;
    int       ste;
    double    elapsed;
} DepthResult;

static DepthResult iter_results[50];

int c_ab_solve_iterative(Board *b, int max_depth, DepthResult *results) {
    tt_clear();
    killers_clear();

    for (int d = 1; d <= max_depth; d++) {
        ab_nodes = 0;
        tt_hits = 0;

        /* Re-setup board for each depth (caller may have modified) */
        ABResult r = ab_solve(b, d, -2.0f, 2.0f, 0);

        results[d - 1].nodes = ab_nodes;
        results[d - 1].tt_hit = tt_hits;
        results[d - 1].value = r.value;
        results[d - 1].ste = r.ste;
        results[d - 1].elapsed = 0;  /* caller times it */
    }
    return max_depth;
}

long long c_get_ab_nodes(void) { return ab_nodes; }
long long c_get_tt_hits(void) { return tt_hits; }

/* ===================================================================
 * Hybrid NN+C Alpha-Beta (like Stockfish NNUE)
 *
 * Uses C for move ordering + pruning (fast) and NN for leaf evaluation.
 * The NN callback is called at depth <= nn_depth.
 * Above nn_depth, uses fast C board_evaluate().
 * =================================================================== */

typedef float (*nn_eval_fn)(void *board_ptr, int current_player, void *ctx);

static nn_eval_fn g_nn_eval = NULL;
static void      *g_nn_ctx  = NULL;
static long long  nn_evals  = 0;

static ABResult nn_ab_solve(Board *b, int depth, int nn_depth,
                             float alpha, float beta, int ply) {
    ABResult result;
    ab_nodes++;

    /* Terminal */
    if (b->winner >= 0) {
        result.value = (b->winner == 0) ? 1.0f : -1.0f;
        result.ste = 0;
        return result;
    }

    /* Leaf: evaluate with NN at shallow depth, C heuristic at deep */
    if (depth <= 0) {
        result.value = board_evaluate(b);  /* always use fast C heuristic at leaves */
        result.ste = 0;
        return result;
    }

    /* At nn_depth boundary: call NN for better evaluation */
    if (depth <= nn_depth && nn_depth > 0 && g_nn_eval) {
        nn_evals++;
        result.value = g_nn_eval(b, b->current_player, g_nn_ctx);
        result.ste = 0;
        return result;
    }

    int p = b->current_player;
    int maximizing = (p == 0);

    /* TT probe */
    uint64_t key = b->zhash;
    TTEntry *tte = tt_probe(key);
    if (tte && tte->depth >= depth) {
        tt_hits++;
        if (tte->flag == 0) { result.value = tte->value; result.ste = tte->ste; return result; }
        if (tte->flag == 1 && tte->value >= beta) { result.value = tte->value; result.ste = tte->ste; return result; }
        if (tte->flag == 2 && tte->value <= alpha) { result.value = tte->value; result.ste = tte->ste; return result; }
    }

    /* Immediate win */
    if (board_has_winning_move(b, p)) {
        result.value = maximizing ? 1.0f : -1.0f;
        result.ste = 1;
        tt_store(key, depth, result.value, result.ste, 0);
        return result;
    }

    /* Double threat */
    if (b->stones_this_turn == 0 && board_count_winning_moves(b, p) >= 3) {
        result.value = maximizing ? 1.0f : -1.0f;
        result.ste = 2;
        tt_store(key, depth, result.value, result.ste, 0);
        return result;
    }

    /* Get moves */
    int move_q[40], move_r[40], move_sc[40];
    int move_limit = (depth <= 3) ? 12 : (depth <= 6 ? 15 : 20);
    int nmoves = board_get_scored_moves(b, move_q, move_r, move_sc, move_limit);
    if (nmoves == 0) { result.value = 0.0f; result.ste = 0; return result; }

    /* Killers */
    if (ply < MAX_PLY) {
        for (int k = 1; k >= 0; k--) {
            int kq = killer_q[ply][k], kr = killer_r[ply][k];
            if (kq == 0 && kr == 0) continue;
            int kqi = kq + OFF, kri = kr + OFF;
            if (kqi >= 0 && kqi < SIZE && kri >= 0 && kri < SIZE &&
                b->board[kqi][kri] == 0 && b->is_cand[kqi][kri]) {
                for (int i = 0; i < nmoves; i++) {
                    if (move_q[i] == kq && move_r[i] == kr) {
                        int target = (k == 0) ? 0 : (nmoves > 1 ? 1 : 0);
                        int tq = move_q[target], tr = move_r[target], ts = move_sc[target];
                        move_q[target] = kq; move_r[target] = kr; move_sc[target] = move_sc[i];
                        move_q[i] = tq; move_r[i] = tr; move_sc[i] = ts;
                        break;
                    }
                }
            }
        }
    }

    float orig_alpha = alpha;
    float best_val = maximizing ? -2.0f : 2.0f;
    int best_ste = 0;
    int best_move_q = move_q[0], best_move_r = move_r[0];

    for (int i = 0; i < nmoves; i++) {
        board_place(b, move_q[i], move_r[i]);

        ABResult child;
        int do_full = 1;
        /* LMR: reduced search for later moves */
        if (i >= 8 && depth >= 4) {
            child = nn_ab_solve(b, depth - 3, nn_depth, alpha, beta, ply + 1);
            if (maximizing && child.value <= alpha) do_full = 0;
            else if (!maximizing && child.value >= beta) do_full = 0;
        }
        if (do_full) {
            child = nn_ab_solve(b, depth - 1, nn_depth, alpha, beta, ply + 1);
        }

        board_undo(b);
        int ste = child.ste + 1;

        if (maximizing) {
            if (child.value > best_val || (child.value == best_val && ste < best_ste)) {
                best_val = child.value; best_ste = ste;
                best_move_q = move_q[i]; best_move_r = move_r[i];
            }
            if (best_val > alpha) alpha = best_val;
        } else {
            if (child.value < best_val || (child.value == best_val && ste > best_ste)) {
                best_val = child.value; best_ste = ste;
                best_move_q = move_q[i]; best_move_r = move_r[i];
            }
            if (best_val < beta) beta = best_val;
        }
        if (alpha >= beta) {
            record_killer(ply, move_q[i], move_r[i]);
            break;
        }
    }

    int flag;
    if (best_val <= orig_alpha) flag = 2;
    else if (best_val >= beta)  flag = 1;
    else                        flag = 0;
    tt_store(key, depth, best_val, best_ste, flag);

    result.value = best_val;
    result.ste = best_ste;
    return result;
}

/* Public API: NN-guided alpha-beta search
 * depth: total search depth
 * nn_depth: only call NN when remaining depth <= nn_depth (0=leaves only, -1=never)
 *   Use nn_depth=0 for NN at leaves, nn_depth=2 for NN at depth 0-2
 *   Use nn_depth=-1 for pure C heuristic (no NN calls — fastest)
 */
long long c_nn_ab_search(Board *b, int depth, int nn_depth,
                          nn_eval_fn eval_fn, void *ctx,
                          float *out_value, int *out_best_q, int *out_best_r,
                          long long *out_nn_evals) {
    ab_nodes = 0;
    tt_hits = 0;
    nn_evals = 0;
    g_nn_eval = eval_fn;
    g_nn_ctx = ctx;

    int p = b->current_player;
    int maximizing = (p == 0);

    int move_q[40], move_r[40], move_sc[40];
    int nmoves = board_get_scored_moves(b, move_q, move_r, move_sc, 25);

    float best_val = maximizing ? -2.0f : 2.0f;
    int best_q = nmoves > 0 ? move_q[0] : 0;
    int best_r = nmoves > 0 ? move_r[0] : 0;

    for (int i = 0; i < nmoves; i++) {
        board_place(b, move_q[i], move_r[i]);
        ABResult child = nn_ab_solve(b, depth - 1, nn_depth, -2.0f, 2.0f, 1);
        board_undo(b);

        if (maximizing) {
            if (child.value > best_val) {
                best_val = child.value;
                best_q = move_q[i];
                best_r = move_r[i];
            }
        } else {
            if (child.value < best_val) {
                best_val = child.value;
                best_q = move_q[i];
                best_r = move_r[i];
            }
        }
    }

    *out_value = best_val;
    *out_best_q = best_q;
    *out_best_r = best_r;
    *out_nn_evals = nn_evals;
    g_nn_eval = NULL;
    g_nn_ctx = NULL;
    return ab_nodes;
}

/* ===================================================================
 * Batched NN Alpha-Beta (collect-inject pattern, zero callbacks)
 *
 * Phase 1: Search with C heuristic, collect leaf positions that need NN eval
 * Phase 2: Python batch-evaluates all leaves with one NN forward pass
 * Phase 3: Re-search with cached NN values
 * =================================================================== */

/* NN Value Cache (separate from TT — different semantics) */
#define NN_CACHE_SIZE (1 << 17)   /* 128K entries */
#define NN_CACHE_MASK (NN_CACHE_SIZE - 1)

typedef struct { uint64_t hash; float value; int valid; } NNCacheEntry;
static NNCacheEntry nn_cache[NN_CACHE_SIZE];

/* Leaf collection buffer */
#define MAX_LEAVES 2048
#define ENC_PLANES 5     /* 5 channels for leaf collection (training uses Python encoding) */
#define ENC_DIM    19
#define ENC_FLOATS (ENC_PLANES * ENC_DIM * ENC_DIM)  /* 5 * 19 * 19 = 1805 */

/* Forward declaration */
void board_encode_state(Board *b, float *output, int *out_offset_q, int *out_offset_r);

static float    leaf_encodings[MAX_LEAVES * ENC_FLOATS];
static int      leaf_players[MAX_LEAVES];
static uint64_t leaf_hashes[MAX_LEAVES];
static int      leaf_count = 0;
static int      batched_collect_mode = 0;  /* 1=collecting, 0=using cache */
static long long batched_nn_hits = 0;

/* --- NN Cache operations --- */

static NNCacheEntry *nn_cache_probe(uint64_t key) {
    NNCacheEntry *e = &nn_cache[key & NN_CACHE_MASK];
    return (e->valid && e->hash == key) ? e : NULL;
}

void c_nn_cache_clear(void) {
    memset(nn_cache, 0, sizeof(nn_cache));
}

void c_nn_cache_inject(uint64_t hash, float value) {
    NNCacheEntry *e = &nn_cache[hash & NN_CACHE_MASK];
    e->hash = hash;
    e->value = value;
    e->valid = 1;
}

void c_nn_cache_inject_batch(uint64_t *hashes, float *values, int count) {
    for (int i = 0; i < count; i++) {
        NNCacheEntry *e = &nn_cache[hashes[i] & NN_CACHE_MASK];
        e->hash = hashes[i];
        e->value = values[i];
        e->valid = 1;
    }
}

/* --- Leaf buffer operations --- */

int   c_get_leaf_count(void)     { return leaf_count; }
float *c_get_leaf_encodings(void) { return leaf_encodings; }
int   *c_get_leaf_players(void)  { return leaf_players; }
uint64_t *c_get_leaf_hashes(void) { return leaf_hashes; }
void  c_set_collect_mode(int m)  { batched_collect_mode = m; }
void  c_clear_leaves(void)       { leaf_count = 0; }
void  c_tt_clear(void)           { memset(tt_table, 0, sizeof(tt_table)); }

/* --- Batched AB solver (no callbacks) --- */

static ABResult batched_ab_solve(Board *b, int depth, int nn_depth,
                                  float alpha, float beta, int ply) {
    ABResult result;
    ab_nodes++;

    /* Terminal */
    if (b->winner >= 0) {
        result.value = (b->winner == 0) ? 1.0f : -1.0f;
        result.ste = 0;
        return result;
    }

    /* Leaf: C heuristic */
    if (depth <= 0) {
        result.value = board_evaluate(b);
        result.ste = 0;
        return result;
    }

    /* At nn_depth boundary: use NN cache or collect */
    if (depth <= nn_depth && nn_depth > 0) {
        /* Check NN cache */
        NNCacheEntry *nne = nn_cache_probe(b->zhash);
        if (nne) {
            batched_nn_hits++;
            result.value = nne->value;
            result.ste = 0;
            return result;
        }
        /* Cache miss */
        if (batched_collect_mode && leaf_count < MAX_LEAVES) {
            /* Encode and store for batch eval */
            int oq, orr;
            board_encode_state(b,
                &leaf_encodings[leaf_count * ENC_FLOATS],
                &oq, &orr);
            leaf_players[leaf_count] = b->current_player;
            leaf_hashes[leaf_count] = b->zhash;
            leaf_count++;
        }
        /* Return C heuristic as placeholder/fallback */
        result.value = board_evaluate(b);
        result.ste = 0;
        return result;
    }

    int p = b->current_player;
    int maximizing = (p == 0);

    /* TT probe */
    uint64_t key = b->zhash;
    TTEntry *tte = tt_probe(key);
    if (tte && tte->depth >= depth) {
        tt_hits++;
        if (tte->flag == 0) { result.value = tte->value; result.ste = tte->ste; return result; }
        if (tte->flag == 1 && tte->value >= beta) { result.value = tte->value; result.ste = tte->ste; return result; }
        if (tte->flag == 2 && tte->value <= alpha) { result.value = tte->value; result.ste = tte->ste; return result; }
    }

    /* Immediate win */
    if (board_has_winning_move(b, p)) {
        result.value = maximizing ? 1.0f : -1.0f;
        result.ste = 1;
        tt_store(key, depth, result.value, result.ste, 0);
        return result;
    }

    /* Double threat */
    if (b->stones_this_turn == 0 && board_count_winning_moves(b, p) >= 3) {
        result.value = maximizing ? 1.0f : -1.0f;
        result.ste = 2;
        tt_store(key, depth, result.value, result.ste, 0);
        return result;
    }

    /* Get moves — tighter limits for deep search */
    int move_q[40], move_r[40], move_sc[40];
    int move_limit = (depth <= 2) ? 8 : (depth <= 4) ? 10 : (depth <= 8) ? 14 : 18;
    int nmoves = board_get_scored_moves(b, move_q, move_r, move_sc, move_limit);
    if (nmoves == 0) { result.value = 0.0f; result.ste = 0; return result; }

    /* Killers: move killer moves to front of list */
    if (ply < MAX_PLY) {
        for (int k = 1; k >= 0; k--) {
            int kq = killer_q[ply][k], kr = killer_r[ply][k];
            if (kq == 0 && kr == 0) continue;
            int kqi = kq + OFF, kri = kr + OFF;
            if (kqi >= 0 && kqi < SIZE && kri >= 0 && kri < SIZE &&
                b->board[kqi][kri] == 0 && b->is_cand[kqi][kri]) {
                for (int i = 0; i < nmoves; i++) {
                    if (move_q[i] == kq && move_r[i] == kr) {
                        int target = (k == 0) ? 0 : (nmoves > 1 ? 1 : 0);
                        int tq = move_q[target], tr = move_r[target], ts = move_sc[target];
                        move_q[target] = kq; move_r[target] = kr; move_sc[target] = move_sc[i];
                        move_q[i] = tq; move_r[i] = tr; move_sc[i] = ts;
                        break;
                    }
                }
            }
        }
    }

    float orig_alpha = alpha;
    float best_val = maximizing ? -2.0f : 2.0f;
    int best_ste = 0;

    for (int i = 0; i < nmoves; i++) {
        board_place(b, move_q[i], move_r[i]);

        ABResult child;
        int do_full = 1;

        /* === ADAPTIVE LATE MOVE REDUCTION ===
         * Later moves get progressively more reduction. */
        if (depth >= 3 && i >= 3) {
            int reduction = (i >= 12) ? 4 : (i >= 6) ? 3 : 2;
            if (depth <= 4) reduction = (reduction > 1) ? reduction - 1 : 1;
            child = batched_ab_solve(b, depth - 1 - reduction, nn_depth,
                                      alpha, beta, ply + 1);
            if (maximizing && child.value <= alpha) do_full = 0;
            else if (!maximizing && child.value >= beta) do_full = 0;
        }
        if (do_full) {
            child = batched_ab_solve(b, depth - 1, nn_depth, alpha, beta, ply + 1);
        }

        board_undo(b);
        int ste = child.ste + 1;

        if (maximizing) {
            if (child.value > best_val || (child.value == best_val && ste < best_ste)) {
                best_val = child.value; best_ste = ste;
            }
            if (best_val > alpha) alpha = best_val;
        } else {
            if (child.value < best_val || (child.value == best_val && ste > best_ste)) {
                best_val = child.value; best_ste = ste;
            }
            if (best_val < beta) beta = best_val;
        }
        if (alpha >= beta) {
            record_killer(ply, move_q[i], move_r[i]);
            break;
        }
    }

    int flag;
    if (best_val <= orig_alpha) flag = 2;
    else if (best_val >= beta)  flag = 1;
    else                        flag = 0;
    tt_store(key, depth, best_val, best_ste, flag);

    result.value = best_val;
    result.ste = best_ste;
    return result;
}

/* Public API: Batched NN alpha-beta (no callbacks) */
long long c_batched_ab_search(Board *b, int depth, int nn_depth,
                               float *out_value, int *out_best_q, int *out_best_r,
                               long long *out_nn_hits) {
    ab_nodes = 0;
    tt_hits = 0;
    batched_nn_hits = 0;

    int p = b->current_player;
    int maximizing = (p == 0);

    /* Root: get top moves */
    int move_q[40], move_r[40], move_sc[40];
    int nmoves = board_get_scored_moves(b, move_q, move_r, move_sc, 25);
    if (nmoves == 0) {
        *out_value = 0.0f; *out_best_q = 0; *out_best_r = 0; *out_nn_hits = 0;
        return 0;
    }

    float best_val = maximizing ? -2.0f : 2.0f;
    int best_q = move_q[0], best_r = move_r[0];
    float alpha = -2.0f, beta = 2.0f;

    for (int i = 0; i < nmoves; i++) {
        board_place(b, move_q[i], move_r[i]);
        ABResult child = batched_ab_solve(b, depth - 1, nn_depth, alpha, beta, 1);
        board_undo(b);

        if (maximizing) {
            if (child.value > best_val) {
                best_val = child.value;
                best_q = move_q[i]; best_r = move_r[i];
            }
            if (best_val > alpha) alpha = best_val;
        } else {
            if (child.value < best_val) {
                best_val = child.value;
                best_q = move_q[i]; best_r = move_r[i];
            }
            if (best_val < beta) beta = best_val;
        }
    }

    *out_value = best_val;
    *out_best_q = best_q;
    *out_best_r = best_r;
    *out_nn_hits = batched_nn_hits;
    return ab_nodes;
}

/* ===================================================================
 * Batch random rollout (entirely in C)
 * =================================================================== */

static uint32_t lcg_next(uint32_t *state) {
    *state = *state * 1103515245u + 12345u;
    return (*state >> 16) & 0x7fff;
}

int board_play_random_game(Board *b, uint32_t *rng_state) {
    while (b->winner < 0 && b->total_stones < b->max_stones) {
        int n = b->cand_count;
        if (n <= 0) break;
        int idx = (int)(lcg_next(rng_state) % (uint32_t)n);
        int qi = b->cand_qi[idx], ri = b->cand_ri[idx];
        board_place(b, qi - OFF, ri - OFF);
    }
    return b->winner;
}

int board_play_heuristic_game(Board *b, uint32_t *rng_state) {
    int mq[10], mr[10], ms[10];
    while (b->winner < 0 && b->total_stones < b->max_stones) {
        int n = b->cand_count;
        if (n <= 0) break;
        int nm = board_get_scored_moves(b, mq, mr, ms, 8);
        if (nm <= 0) break;
        int idx = (int)(lcg_next(rng_state) % (uint32_t)nm);
        board_place(b, mq[idx], mr[idx]);
    }
    return b->winner;
}

void batch_rollout(int dq1, int dr1, int dq2, int dr2,
                   int num_games, int mode,
                   int *out_x_wins, int *out_o_wins, int *out_draws,
                   unsigned int seed) {
    Board b;
    uint32_t rng = seed;
    int xw = 0, ow = 0, dr = 0;

    for (int g = 0; g < num_games; g++) {
        board_setup_triangle(&b);
        b.max_stones = 150;
        board_place(&b, dq1, dr1);
        board_place(&b, dq2, dr2);

        int winner;
        if (mode == 0) {
            winner = board_play_random_game(&b, &rng);
        } else {
            winner = board_play_heuristic_game(&b, &rng);
        }

        if (winner == 0) xw++;
        else if (winner == 1) ow++;
        else dr++;
    }
    *out_x_wins = xw;
    *out_o_wins = ow;
    *out_draws = dr;
}

/* ===================================================================
 * Accessors for Python ctypes
 * =================================================================== */

int board_get_winner(Board *b)         { return b->winner; }
int board_get_current_player(Board *b) { return b->current_player; }
int board_get_total_stones(Board *b)   { return b->total_stones; }
int board_get_cand_count(Board *b)     { return b->cand_count; }
uint64_t board_get_zhash(Board *b)     { return b->zhash; }
int board_get_stones_this_turn(Board *b) { return b->stones_this_turn; }

int board_sizeof(void) { return (int)sizeof(Board); }

/* ===================================================================
 * Greedy self-play tournament (entirely in C)
 *
 * Both players pick from top-K scored moves with temperature control.
 * Records every move for analysis by Python.
 * =================================================================== */

#define TOURNEY_MAX_MOVES 150

typedef struct {
    int16_t move_q[TOURNEY_MAX_MOVES];
    int16_t move_r[TOURNEY_MAX_MOVES];
    int16_t move_count;
    int8_t  winner;   /* 0=X, 1=O, -1=draw */
} GameRecord;

void greedy_tournament(
    int num_games,
    int top_k,             /* select from top-K scored moves */
    int deterministic,     /* 1 = always pick best, 0 = weighted random among top-K */
    GameRecord *records,   /* pre-allocated output: records[num_games] */
    unsigned int seed
) {
    Board b;
    uint32_t rng = seed;
    int mq[40], mr[40], ms[40];

    for (int g = 0; g < num_games; g++) {
        board_setup_triangle(&b);
        b.max_stones = TOURNEY_MAX_MOVES;

        GameRecord *rec = &records[g];
        rec->move_count = 0;
        rec->winner = -1;

        while (b.winner < 0 && b.total_stones < b.max_stones && rec->move_count < TOURNEY_MAX_MOVES) {
            int n = b.cand_count;
            if (n <= 0) break;

            int nm = board_get_scored_moves(&b, mq, mr, ms, top_k);
            if (nm <= 0) break;

            int idx;
            if (deterministic || nm == 1) {
                idx = 0;  /* best move */
            } else {
                /* Softmax-like weighted random among top-K */
                /* Simple approach: weight by score rank (higher rank = more likely) */
                int total_weight = 0;
                int weights[40];
                for (int i = 0; i < nm; i++) {
                    weights[i] = nm - i;  /* rank weight: best=nm, worst=1 */
                    total_weight += weights[i];
                }
                int pick = (int)(lcg_next(&rng) % (uint32_t)total_weight);
                idx = 0;
                int acc = 0;
                for (int i = 0; i < nm; i++) {
                    acc += weights[i];
                    if (pick < acc) { idx = i; break; }
                }
            }

            rec->move_q[rec->move_count] = (int16_t)mq[idx];
            rec->move_r[rec->move_count] = (int16_t)mr[idx];
            rec->move_count++;

            board_place(&b, mq[idx], mr[idx]);
        }

        rec->winner = (int8_t)b.winner;
    }
}

int game_record_sizeof(void) { return (int)sizeof(GameRecord); }
int game_record_max_moves(void) { return TOURNEY_MAX_MOVES; }

/* ===================================================================
 * THREAT-ONLY SOLVER — mimics human case analysis
 *
 * X: only considers moves that create/extend threats (3+ on axis)
 * O: only considers moves that block X's strongest threats
 *
 * Branching factor ~3-5 instead of ~15, enabling depth 30+
 * =================================================================== */

static long long threat_nodes = 0;

/* Simple scored move for threat solver (avoid dependency on later ScoredMove) */
typedef struct { int q, r, score; } ThreatMove;
static int cmp_threat_desc(const void *a, const void *b) {
    return ((ThreatMove*)b)->score - ((ThreatMove*)a)->score;
}

/* Get X's attacking moves: cells that create 3+ on any axis for X */
static int get_x_attacks(Board *b, int *out_q, int *out_r, int max_n) {
    static const int axes[3][2] = {{1,0},{0,1},{1,-1}};
    ThreatMove moves[200];
    int nmoves = 0;

    for (int i = 0; i < b->cand_count && nmoves < 200; i++) {
        int qi = b->cand_qi[i], ri = b->cand_ri[i];
        int score = 0;
        int best_line = 0;

        for (int a = 0; a < 3; a++) {
            int dq = axes[a][0], dr = axes[a][1];
            int c = 1 + count_line_dir(b, qi, ri, dq, dr, 0)
                      + count_line_dir(b, qi, ri, -dq, -dr, 0);
            if (c > best_line) best_line = c;
            if (c >= 3) score += c * c * 100;
        }

        if (best_line >= 6) score = 1000000;
        else if (best_line >= 5) score = 500000;
        else if (best_line >= 4) score = 100000;

        /* Also block O's threats */
        for (int a = 0; a < 3; a++) {
            int dq = axes[a][0], dr = axes[a][1];
            int c = 1 + count_line_dir(b, qi, ri, dq, dr, 1)
                      + count_line_dir(b, qi, ri, -dq, -dr, 1);
            if (c >= 5) score += 200000;
            else if (c >= 4) score += 10000;
        }

        if (score > 0) {
            moves[nmoves].q = qi - OFF;
            moves[nmoves].r = ri - OFF;
            moves[nmoves].score = score;
            nmoves++;
        }
    }

    qsort(moves, nmoves, sizeof(ThreatMove), cmp_threat_desc);
    int count = nmoves < max_n ? nmoves : max_n;
    for (int i = 0; i < count; i++) {
        out_q[i] = moves[i].q;
        out_r[i] = moves[i].r;
    }
    return count;
}

/* Get O's blocking moves: cells that break X's strongest lines */
static int get_o_blocks(Board *b, int *out_q, int *out_r, int max_n) {
    static const int axes[3][2] = {{1,0},{0,1},{1,-1}};
    ThreatMove moves[200];
    int nmoves = 0;

    for (int i = 0; i < b->cand_count && nmoves < 200; i++) {
        int qi = b->cand_qi[i], ri = b->cand_ri[i];
        int score = 0;

        for (int a = 0; a < 3; a++) {
            int dq = axes[a][0], dr = axes[a][1];
            int cx = 1 + count_line_dir(b, qi, ri, dq, dr, 0)
                       + count_line_dir(b, qi, ri, -dq, -dr, 0);
            if (cx >= 6) score += 1000000;
            else if (cx >= 5) score += 500000;
            else if (cx >= 4) score += 50000;
            else if (cx >= 3) score += 5000;

            int co = 1 + count_line_dir(b, qi, ri, dq, dr, 1)
                       + count_line_dir(b, qi, ri, -dq, -dr, 1);
            if (co >= 6) score += 900000;
            else if (co >= 5) score += 400000;
            else if (co >= 4) score += 40000;
            else if (co >= 3) score += 3000;
        }

        if (score > 0) {
            moves[nmoves].q = qi - OFF;
            moves[nmoves].r = ri - OFF;
            moves[nmoves].score = score;
            nmoves++;
        }
    }

    qsort(moves, nmoves, sizeof(ThreatMove), cmp_threat_desc);
    int count = nmoves < max_n ? nmoves : max_n;
    for (int i = 0; i < count; i++) {
        out_q[i] = moves[i].q;
        out_r[i] = moves[i].r;
    }
    return count;
}

static ABResult threat_solve(Board *b, int depth, float alpha, float beta, int ply) {
    ABResult result;
    threat_nodes++;

    /* Terminal */
    if (b->winner >= 0) {
        result.value = (b->winner == 0) ? 1.0f : -1.0f;
        result.ste = 0;
        return result;
    }
    if (depth <= 0) {
        result.value = 0.0f;
        result.ste = 0;
        return result;
    }

    int p = b->current_player;
    int maximizing = (p == 0);

    /* Immediate win */
    if (board_has_winning_move(b, p)) {
        result.value = maximizing ? 1.0f : -1.0f;
        result.ste = 1;
        return result;
    }

    /* Double threat */
    if (b->stones_this_turn == 0) {
        int wm = board_count_winning_moves(b, p);
        if (wm >= 3) {
            result.value = maximizing ? 1.0f : -1.0f;
            result.ste = 2;
            return result;
        }
    }

    /* Get threat-relevant moves only */
    int mq[30], mr[30];
    int nmoves;
    if (p == 0) {
        nmoves = get_x_attacks(b, mq, mr, 8);
    } else {
        nmoves = get_o_blocks(b, mq, mr, 8);
    }

    /* If no threat moves found, try general top moves (limited) */
    if (nmoves == 0) {
        int ms[10];
        nmoves = board_get_scored_moves(b, mq, mr, ms, 5);
    }

    if (nmoves == 0) {
        result.value = 0.0f;
        result.ste = 0;
        return result;
    }

    float orig_alpha = alpha;
    float best_val = maximizing ? -2.0f : 2.0f;
    int best_ste = 0;

    for (int i = 0; i < nmoves; i++) {
        board_place(b, mq[i], mr[i]);
        ABResult child = threat_solve(b, depth - 1, alpha, beta, ply + 1);
        board_undo(b);

        int ste = child.ste + 1;
        if (maximizing) {
            if (child.value > best_val || (child.value == best_val && ste < best_ste)) {
                best_val = child.value;
                best_ste = ste;
            }
            if (best_val > alpha) alpha = best_val;
        } else {
            if (child.value < best_val || (child.value == best_val && ste > best_ste)) {
                best_val = child.value;
                best_ste = ste;
            }
            if (best_val < beta) beta = best_val;
        }
        if (alpha >= beta) break;
    }

    result.value = best_val;
    result.ste = best_ste;
    return result;
}

/* Public API */
long long c_threat_solve(Board *b, int depth, float *out_value, int *out_ste) {
    threat_nodes = 0;
    ABResult r = threat_solve(b, depth, -2.0f, 2.0f, 0);
    *out_value = r.value;
    *out_ste = r.ste;
    return threat_nodes;
}

long long c_get_threat_nodes(void) { return threat_nodes; }

/* ===================================================================
 * PROOF-NUMBER SEARCH (PNS) — entirely in C
 *
 * Focused on proving X wins from triangle + O defense positions.
 * Uses:
 *   1. Proof/disproof number propagation
 *   2. Threat-space filtering (only forcing moves first)
 *   3. Existing TT for dedup
 *   4. Iterative deepening PNS (df-pn variant)
 * =================================================================== */

/* PNS TT entry — separate from alpha-beta TT */
typedef struct {
    uint64_t key;
    uint32_t pn;   /* proof number */
    uint32_t dn;   /* disproof number */
    uint8_t  type; /* 0=unsolved, 1=proven(X wins), 2=disproven(O wins/draw) */
} PNSEntry;

#define PNS_TT_SIZE (1 << 23)  /* 8M entries */
#define PNS_TT_MASK (PNS_TT_SIZE - 1)
#define PNS_INF 0xFFFFFFu      /* infinity for pn/dn */
static PNSEntry pns_tt[PNS_TT_SIZE];
static long long pns_nodes = 0;

static void pns_tt_clear(void) {
    memset(pns_tt, 0, sizeof(pns_tt));
}

static PNSEntry *pns_tt_probe(uint64_t key) {
    PNSEntry *e = &pns_tt[key & PNS_TT_MASK];
    if (e->key == key) return e;
    return NULL;
}

static void pns_tt_store(uint64_t key, uint32_t pn, uint32_t dn, uint8_t type) {
    PNSEntry *e = &pns_tt[key & PNS_TT_MASK];
    e->key = key;
    e->pn = pn;
    e->dn = dn;
    e->type = type;
}

/* ===================================================================
 * Threat-space move generator
 *
 * Returns moves in priority order:
 *   1. Immediate wins (completing 6-in-a-row)
 *   2. Must-blocks (opponent has 5-in-a-row threat)
 *   3. Attack threats (creating 4+ for self)
 *   4. Block threats (blocking opponent's 4+)
 *   5. Top scored non-forcing moves (limited)
 * =================================================================== */

typedef struct {
    int q, r;
    int priority;  /* higher = better */
} ScoredMove;

static int cmp_scored_desc(const void *a, const void *b) {
    return ((ScoredMove*)b)->priority - ((ScoredMove*)a)->priority;
}

static int get_threat_moves(Board *b, int *out_q, int *out_r, int max_moves) {
    int p = b->current_player;
    int opp = 1 - p;
    ScoredMove moves[600];
    int nmoves = 0;

    for (int i = 0; i < b->cand_count && nmoves < 600; i++) {
        int qi = b->cand_qi[i], ri = b->cand_ri[i];
        int prio = 0;

        /* Check if this move wins immediately */
        if (board_would_win(b, qi, ri, p)) {
            prio = 1000000;
        } else {
            /* Check if this blocks opponent's immediate win */
            if (board_would_win(b, qi, ri, opp)) {
                prio = 500000;
            }

            /* Check own line lengths */
            int best_own = 0, best_opp_line = 0;
            int pv = p + 1, ov = opp + 1;
            static const int axes[3][2] = {{1,0},{0,1},{1,-1}};
            for (int a = 0; a < 3; a++) {
                int dq = axes[a][0], dr = axes[a][1];
                int c = 1 + count_line_dir(b, qi, ri, dq, dr, p)
                          + count_line_dir(b, qi, ri, -dq, -dr, p);
                if (c > best_own) best_own = c;

                c = 1 + count_line_dir(b, qi, ri, dq, dr, opp)
                      + count_line_dir(b, qi, ri, -dq, -dr, opp);
                if (c > best_opp_line) best_opp_line = c;
            }

            if (prio < 500000) {
                if (best_own >= 5) prio = 100000;
                else if (best_own >= 4) prio = 10000;
                else if (best_own >= 3) prio = 1000;

                if (best_opp_line >= 5) prio += 80000;
                else if (best_opp_line >= 4) prio += 5000;
                else if (best_opp_line >= 3) prio += 500;

                prio += best_own * 10 + best_opp_line * 5;
            }
        }

        /* Proximity bonus */
        int dq = qi - OFF, dr = ri - OFF;
        prio -= (dq > 0 ? dq : -dq) + (dr > 0 ? dr : -dr);

        moves[nmoves].q = qi - OFF;
        moves[nmoves].r = ri - OFF;
        moves[nmoves].priority = prio;
        nmoves++;
    }

    /* Sort by priority descending */
    qsort(moves, nmoves, sizeof(ScoredMove), cmp_scored_desc);

    /* Return top moves */
    int count = nmoves < max_moves ? nmoves : max_moves;
    for (int i = 0; i < count; i++) {
        out_q[i] = moves[i].q;
        out_r[i] = moves[i].r;
    }
    return count;
}

/* ===================================================================
 * Depth-First Proof-Number Search (df-pn)
 *
 * For each node:
 *   - OR nodes (X to move): pn = min(child pn), dn = sum(child dn)
 *   - AND nodes (O to move): pn = sum(child pn), dn = min(child dn)
 *
 * A node is PROVEN when pn=0, DISPROVEN when dn=0.
 * =================================================================== */

static long long pns_node_limit = 50000000LL;  /* 50M default */
static int pns_recursion_depth = 0;
static int pns_max_recursion = 500;

static void dfpn_solve(Board *b, uint32_t *pn, uint32_t *dn,
                        uint32_t pn_threshold, uint32_t dn_threshold,
                        int max_depth) {
    pns_nodes++;
    pns_recursion_depth++;

    /* Safety limits */
    if (pns_nodes >= pns_node_limit || pns_recursion_depth > pns_max_recursion) {
        *pn = 1; *dn = 1;
        pns_recursion_depth--;
        return;
    }

    /* Terminal check */
    if (b->winner >= 0) {
        if (b->winner == 0) { *pn = 0; *dn = PNS_INF; }
        else                { *pn = PNS_INF; *dn = 0; }
        goto done;
    }
    if (max_depth <= 0) {
        *pn = 1; *dn = 1;
        goto done;
    }

    /* TT lookup */
    {
    uint64_t key = b->zhash;
    PNSEntry *tte = pns_tt_probe(key);
    if (tte) {
        if (tte->type == 1) { *pn = 0; *dn = PNS_INF; goto done; }
        if (tte->type == 2) { *pn = PNS_INF; *dn = 0; goto done; }
        *pn = tte->pn;
        *dn = tte->dn;
        if (*pn >= pn_threshold || *dn >= dn_threshold) goto done;
    }

    int p = b->current_player;
    int is_or = (p == 0);  /* X is the prover (OR node) */

    /* Quick win/threat checks */
    if (board_has_winning_move(b, p)) {
        if (is_or) { *pn = 0; *dn = PNS_INF; }
        else       { *pn = PNS_INF; *dn = 0; }
        pns_tt_store(key, *pn, *dn, is_or ? 1 : 2);
        goto done;
    }

    /* Double-threat at turn start */
    if (b->stones_this_turn == 0) {
        int wm = board_count_winning_moves(b, p);
        if (wm >= 3) {
            if (is_or) { *pn = 0; *dn = PNS_INF; }
            else       { *pn = PNS_INF; *dn = 0; }
            pns_tt_store(key, *pn, *dn, is_or ? 1 : 2);
            goto done;
        }
    }

    /* Generate moves (threat-space ordered) */
    int mq[50], mr[50];
    int nmoves = get_threat_moves(b, mq, mr, 40);
    if (nmoves == 0) {
        *pn = 1; *dn = 1;
        goto done;
    }

    /* df-pn main loop */
    uint32_t child_pn[50], child_dn[50];
    for (int i = 0; i < nmoves; i++) {
        child_pn[i] = 1;
        child_dn[i] = 1;
    }

    for (int iteration = 0; iteration < 100000; iteration++) {
        /* Node limit re-check inside loop */
        if (pns_nodes >= pns_node_limit) {
            *pn = 1; *dn = 1;
            goto done;
        }

        /* Compute current pn/dn from children */
        uint32_t cur_pn, cur_dn;
        if (is_or) {
            cur_pn = PNS_INF;
            cur_dn = 0;
            for (int i = 0; i < nmoves; i++) {
                if (child_pn[i] < cur_pn) cur_pn = child_pn[i];
                uint32_t new_dn = cur_dn + child_dn[i];
                cur_dn = (new_dn < cur_dn) ? PNS_INF : new_dn;
            }
        } else {
            cur_pn = 0;
            cur_dn = PNS_INF;
            for (int i = 0; i < nmoves; i++) {
                uint32_t new_pn = cur_pn + child_pn[i];
                cur_pn = (new_pn < cur_pn) ? PNS_INF : new_pn;
                if (child_dn[i] < cur_dn) cur_dn = child_dn[i];
            }
        }

        /* Check thresholds */
        if (cur_pn >= pn_threshold || cur_dn >= dn_threshold ||
            cur_pn == 0 || cur_dn == 0) {
            *pn = cur_pn;
            *dn = cur_dn;
            uint8_t type = 0;
            if (cur_pn == 0) type = 1;
            else if (cur_dn == 0) type = 2;
            pns_tt_store(key, cur_pn, cur_dn, type);
            goto done;
        }

        /* Select most proving child */
        int best_idx = 0;
        if (is_or) {
            for (int i = 1; i < nmoves; i++)
                if (child_pn[i] < child_pn[best_idx]) best_idx = i;
        } else {
            for (int i = 1; i < nmoves; i++)
                if (child_dn[i] < child_dn[best_idx]) best_idx = i;
        }

        /* Compute child thresholds */
        uint32_t child_pn_thr, child_dn_thr;
        if (is_or) {
            uint32_t second_pn = PNS_INF;
            for (int i = 0; i < nmoves; i++)
                if (i != best_idx && child_pn[i] < second_pn)
                    second_pn = child_pn[i];
            child_pn_thr = pn_threshold < second_pn + 1 ? pn_threshold : second_pn + 1;
            child_dn_thr = dn_threshold - cur_dn + child_dn[best_idx];
        } else {
            uint32_t second_dn = PNS_INF;
            for (int i = 0; i < nmoves; i++)
                if (i != best_idx && child_dn[i] < second_dn)
                    second_dn = child_dn[i];
            child_dn_thr = dn_threshold < second_dn + 1 ? dn_threshold : second_dn + 1;
            child_pn_thr = pn_threshold - cur_pn + child_pn[best_idx];
        }

        /* Recurse into best child */
        board_place(b, mq[best_idx], mr[best_idx]);
        uint32_t cpn, cdn;
        dfpn_solve(b, &cpn, &cdn, child_pn_thr, child_dn_thr, max_depth - 1);
        board_undo(b);

        child_pn[best_idx] = cpn;
        child_dn[best_idx] = cdn;
    }

    /* Iteration limit reached */
    *pn = 1; *dn = 1;
    }  /* end of scope for key, tte, etc. */

done:
    pns_recursion_depth--;
}

/* Public API: PNS solve */
long long c_pns_solve(Board *b, int max_depth, long long node_limit, int *out_result) {
    pns_nodes = 0;
    pns_node_limit = node_limit > 0 ? node_limit : 50000000LL;
    pns_tt_clear();

    uint32_t pn = 1, dn = 1;
    dfpn_solve(b, &pn, &dn, PNS_INF, PNS_INF, max_depth);

    if (pn == 0) *out_result = 1;       /* X wins (proven) */
    else if (dn == 0) *out_result = -1;  /* O wins/draws (disproven) */
    else *out_result = 0;                /* inconclusive */

    return pns_nodes;
}

long long c_get_pns_nodes(void) { return pns_nodes; }

/* ===================================================================
 * Comprehensive defense scanner
 *
 * Tests ALL possible O defense pairs from A+B ring, B+B ring, etc.
 * For each pair, runs:
 *   1. Quick heuristic tournament (N games)
 *   2. Alpha-beta to specified depth
 *   3. Optionally PNS
 *
 * Results sorted by O win rate.
 * =================================================================== */

typedef struct {
    int dq1, dr1, dq2, dr2;   /* O's defense moves (axial coords) */
    int games_played;
    int x_wins, o_wins, draws;
    float ab_value;            /* alpha-beta result */
    int ab_ste;
    int ab_depth;
    int pns_result;            /* 1=X proven, -1=O proven, 0=unclear */
    long long pns_nodes;
} DefenseResult;

/* Pre-computed ring positions */
static const int A_RING_Q[] = {1, 0, -1, -1, 0, 1};
static const int A_RING_R[] = {0, 1, 1, 0, -1, -1};
static const int B_RING_Q[] = {2, 1, 0, -1, -2, -2, -2, -1, 0, 1, 2, 2};
static const int B_RING_R[] = {0, 1, 2, 2, 2, 1, 0, -1, -2, -2, -2, -1};
static const int C_RING_Q[] = {3, 2, 1, 0, -1, -2, -3, -3, -3, -3, -2, -1, 0, 1, 2, 3, 3, 3};
static const int C_RING_R[] = {0, 1, 2, 3, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -3, -2, -1};

int c_scan_defenses(
    int num_games,       /* tournament games per defense */
    int ab_depth,        /* alpha-beta depth */
    int use_pns,         /* 0=skip PNS, 1=run PNS */
    int pns_depth,       /* PNS max depth */
    DefenseResult *results,  /* pre-allocated output */
    int max_results      /* max results to fill */
) {
    /* Build list of all valid defense pairs:
     * - Both cells from A+B+C rings
     * - Exclude cells occupied by triangle: (0,0), (1,0), (0,1)
     * - Only pairs where both cells are distinct and unoccupied
     */
    int all_q[36], all_r[36];
    int n_cells = 0;

    /* A-ring (skip A0=(1,0) and A1=(0,1) — occupied by triangle) */
    for (int i = 0; i < 6; i++) {
        if (A_RING_Q[i] == 1 && A_RING_R[i] == 0) continue;  /* A0 = triangle */
        if (A_RING_Q[i] == 0 && A_RING_R[i] == 1) continue;  /* A1 = triangle */
        all_q[n_cells] = A_RING_Q[i];
        all_r[n_cells] = A_RING_R[i];
        n_cells++;
    }

    /* B-ring (all 12) */
    for (int i = 0; i < 12; i++) {
        all_q[n_cells] = B_RING_Q[i];
        all_r[n_cells] = B_RING_R[i];
        n_cells++;
    }

    int count = 0;
    Board b;
    uint32_t rng = 42;

    for (int i = 0; i < n_cells && count < max_results; i++) {
        for (int j = i + 1; j < n_cells && count < max_results; j++) {
            DefenseResult *res = &results[count];
            res->dq1 = all_q[i]; res->dr1 = all_r[i];
            res->dq2 = all_q[j]; res->dr2 = all_r[j];

            /* Tournament */
            int xw = 0, ow = 0, dr = 0;
            for (int g = 0; g < num_games; g++) {
                board_setup_triangle(&b);
                b.max_stones = 150;
                board_place(&b, all_q[i], all_r[i]);
                board_place(&b, all_q[j], all_r[j]);
                int w = board_play_heuristic_game(&b, &rng);
                if (w == 0) xw++;
                else if (w == 1) ow++;
                else dr++;
            }
            res->games_played = num_games;
            res->x_wins = xw;
            res->o_wins = ow;
            res->draws = dr;

            /* Alpha-beta */
            board_setup_triangle(&b);
            board_place(&b, all_q[i], all_r[i]);
            board_place(&b, all_q[j], all_r[j]);

            float val = 0;
            int ste = 0;
            tt_clear();
            killers_clear();
            ab_nodes = 0;
            ABResult abr = ab_solve(&b, ab_depth, -2.0f, 2.0f, 0);
            res->ab_value = abr.value;
            res->ab_ste = abr.ste;
            res->ab_depth = ab_depth;

            /* PNS (optional) */
            res->pns_result = 0;
            res->pns_nodes = 0;
            if (use_pns) {
                board_setup_triangle(&b);
                board_place(&b, all_q[i], all_r[i]);
                board_place(&b, all_q[j], all_r[j]);

                int pns_res = 0;
                long long pn = c_pns_solve(&b, pns_depth, 10000000LL, &pns_res);
                res->pns_result = pns_res;
                res->pns_nodes = pn;
            }

            count++;
        }
    }

    return count;
}

int defense_result_sizeof(void) { return (int)sizeof(DefenseResult); }

/* ===================================================================
 * Training integration: state encoding, legal mask, threat label
 * =================================================================== */

#define NN_SIZE 19
#define NN_HALF 9
#define NN_PLANES 5  /* base encoding; Python adds threat channels 5-6 */

void board_encode_state(Board *b, float *output, int *out_offset_q, int *out_offset_r) {
    /*
     * Encode board as (5, 19, 19) float tensor matching bot.py encode_state().
     * Plane 0: current player stones
     * Plane 1: opponent stones
     * Plane 2: legal moves (candidates)
     * Plane 3: current player indicator
     * Plane 4: stones remaining this turn (normalized)
     */
    int total = NN_PLANES * NN_SIZE * NN_SIZE;
    memset(output, 0, total * sizeof(float));

    int p = b->current_player;
    int cur_val = p + 1;       /* board value for current player */
    int opp_val = 2 - p;       /* board value for opponent */

    /* Compute centroid of occupied cells */
    int sum_q = 0, sum_r = 0, count = 0;
    for (int qi = 0; qi < SIZE; qi++) {
        for (int ri = 0; ri < SIZE; ri++) {
            if (b->board[qi][ri] != 0) {
                sum_q += (qi - OFF);
                sum_r += (ri - OFF);
                count++;
            }
        }
    }

    int cq = 0, cr = 0;
    if (count > 0) {
        /* Round to nearest integer (matching Python round()) */
        cq = (sum_q >= 0) ? (sum_q + count/2) / count : (sum_q - count/2) / count;
        cr = (sum_r >= 0) ? (sum_r + count/2) / count : (sum_r - count/2) / count;
    }
    int offset_q = cq - NN_HALF;
    int offset_r = cr - NN_HALF;
    *out_offset_q = offset_q;
    *out_offset_r = offset_r;

    /* Plane 0: current player stones */
    float *p0 = output;
    /* Plane 1: opponent stones */
    float *p1 = output + NN_SIZE * NN_SIZE;

    for (int qi = 0; qi < SIZE; qi++) {
        for (int ri = 0; ri < SIZE; ri++) {
            int v = b->board[qi][ri];
            if (v == 0) continue;
            int q = qi - OFF, r = ri - OFF;
            int i = q - offset_q, j = r - offset_r;
            if (i >= 0 && i < NN_SIZE && j >= 0 && j < NN_SIZE) {
                if (v == cur_val)
                    p0[i * NN_SIZE + j] = 1.0f;
                else
                    p1[i * NN_SIZE + j] = 1.0f;
            }
        }
    }

    /* Plane 2: legal moves (candidates) */
    float *p2 = output + 2 * NN_SIZE * NN_SIZE;
    if (count == 0) {
        /* First move: only (0,0) */
        int i = -offset_q, j = -offset_r;
        if (i >= 0 && i < NN_SIZE && j >= 0 && j < NN_SIZE)
            p2[i * NN_SIZE + j] = 1.0f;
    } else {
        for (int c = 0; c < b->cand_count; c++) {
            int q = b->cand_qi[c] - OFF;
            int r = b->cand_ri[c] - OFF;
            int i = q - offset_q, j = r - offset_r;
            if (i >= 0 && i < NN_SIZE && j >= 0 && j < NN_SIZE)
                p2[i * NN_SIZE + j] = 1.0f;
        }
    }

    /* Plane 3: current player indicator */
    float *p3 = output + 3 * NN_SIZE * NN_SIZE;
    float player_val = (float)p;
    for (int k = 0; k < NN_SIZE * NN_SIZE; k++)
        p3[k] = player_val;

    /* Plane 4: stones remaining this turn (normalized) */
    float *p4 = output + 4 * NN_SIZE * NN_SIZE;
    float remaining = (float)(b->stones_per_turn - b->stones_this_turn) / 2.0f;
    for (int k = 0; k < NN_SIZE * NN_SIZE; k++)
        p4[k] = remaining;

    /* Threat channels (5-6) are added by Python's c_encode_state() */
}

int board_get_legal_mask(Board *b, float *mask, int offset_q, int offset_r) {
    /*
     * Fill float[361] with 1.0 for legal moves within 19x19 window.
     * Returns count of legal moves in window.
     */
    memset(mask, 0, NN_SIZE * NN_SIZE * sizeof(float));
    int count = 0;

    if (b->total_stones == 0) {
        /* First move: only (0,0) */
        int i = -offset_q, j = -offset_r;
        if (i >= 0 && i < NN_SIZE && j >= 0 && j < NN_SIZE) {
            mask[i * NN_SIZE + j] = 1.0f;
            count = 1;
        }
        return count;
    }

    for (int c = 0; c < b->cand_count; c++) {
        int q = b->cand_qi[c] - OFF;
        int r = b->cand_ri[c] - OFF;
        int i = q - offset_q, j = r - offset_r;
        if (i >= 0 && i < NN_SIZE && j >= 0 && j < NN_SIZE) {
            mask[i * NN_SIZE + j] = 1.0f;
            count++;
        }
    }
    return count;
}

void board_compute_threat_label(Board *b, float *output) {
    /*
     * output[0] = has_4_plus (current player)
     * output[1] = has_5_plus (current player)
     * output[2] = opp_has_4_plus
     * output[3] = opp_has_5_plus
     */
    int p = b->current_player;
    int opp = 1 - p;
    int my_max = 0, opp_max = 0;

    /* Scan all occupied cells for max line lengths */
    for (int qi = 0; qi < SIZE; qi++) {
        for (int ri = 0; ri < SIZE; ri++) {
            int v = b->board[qi][ri];
            if (v == 0) continue;
            int stone_player = v - 1;
            int ml = board_max_line_through(b, qi, ri, stone_player);
            if (stone_player == p && ml > my_max) my_max = ml;
            if (stone_player == opp && ml > opp_max) opp_max = ml;
        }
    }

    output[0] = (my_max >= 4) ? 1.0f : 0.0f;
    output[1] = (my_max >= 5) ? 1.0f : 0.0f;
    output[2] = (opp_max >= 4) ? 1.0f : 0.0f;
    output[3] = (opp_max >= 5) ? 1.0f : 0.0f;
}

void board_copy(Board *dst, Board *src) {
    memcpy(dst, src, sizeof(Board));
}

int board_get_stones_per_turn(Board *b) { return b->stones_per_turn; }
