#include <mpi.h>
#include <array>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <unordered_map>
#include <deque>
#include <regex>
#include <cctype>

using std::string;

// -------------------- Types & helpers --------------------
struct M2 { double v[2][2]{}; };

static inline M2 add2(const M2& A, const M2& B, double a=1.0, double b=1.0){
    M2 R{};
    for(int i=0;i<2;i++)
        for(int j=0;j<2;j++)
            R.v[i][j] = a*A.v[i][j] + b*B.v[i][j];
    return R;
}
static inline M2 mul2(const M2& A, const M2& B){
    M2 R{};
    R.v[0][0] = A.v[0][0]*B.v[0][0] + A.v[0][1]*B.v[1][0];
    R.v[0][1] = A.v[0][0]*B.v[0][1] + A.v[0][1]*B.v[1][1];
    R.v[1][0] = A.v[1][0]*B.v[0][0] + A.v[1][1]*B.v[1][0];
    R.v[1][1] = A.v[1][0]*B.v[0][1] + A.v[1][1]*B.v[1][1];
    return R;
}
static inline void pack_M2(const M2& m, double buf[4]){
    buf[0]=m.v[0][0]; buf[1]=m.v[0][1];
    buf[2]=m.v[1][0]; buf[3]=m.v[1][1];
}
static inline M2 unpack_M2(const double buf[4]){
    M2 m{};
    m.v[0][0]=buf[0]; m.v[0][1]=buf[1];
    m.v[1][0]=buf[2]; m.v[1][1]=buf[3];
    return m;
}
static inline void stitch8(double A[8][8], const std::array<M2,16>& blocks){
    auto idx = [&](int r2,int c2){ return r2*4 + c2; };
    for(int r2=0;r2<4;r2++){
        for(int c2=0;c2<4;c2++){
            const M2& b = blocks[idx(r2,c2)];
            for(int i=0;i<2;i++)
                for(int j=0;j<2;j++)
                    A[r2*2+i][c2*2+j] = b.v[i][j];
        }
    }
}

// -------------------- Leaf recipe representation --------------------
struct Term { int id; int8_t s; };     // s ∈ {+1,-1}
struct Leaf {
    std::vector<Term> Aterms;          // ids in [0..15]
    std::vector<Term> Bterms;          // ids in [16..31]
    std::vector<Term> Couts;           // C ids in [0..15] with sign
};

// -------------------- Owner store and placement --------------------
struct Store { std::map<int,M2> data; }; // id -> 2x2
static std::array<int,32> OWNER;          // global owner table 0..31
static inline int owner_of(int id, int /*world*/) { return OWNER[id]; }

// -------------------- Simple placement loader (49 lines, two tokens) --------------------
// Accepts: a11 / b24 OR A:a11 / B:b24 (case-insensitive). Comments/blank lines ignored but do not count toward 49.
static inline int sym_to_id_simple(const std::string& tok){
    size_t p = tok.find(':');
    std::string s = (p==std::string::npos) ? tok : tok.substr(p+1);
    if (s.size() < 3) throw std::runtime_error("Bad token: "+tok);
    char ab = std::tolower(s[0]);
    if (ab!='a' && ab!='b') throw std::runtime_error("Bad token prefix (need a/b): "+tok);
    if (!std::isdigit((unsigned char)s[1]) || !std::isdigit((unsigned char)s[2]))
        throw std::runtime_error("Bad indices in token: "+tok);
    int r = s[1]-'0', c = s[2]-'0';
    if (r<1||r>4||c<1||c>4) throw std::runtime_error("Out-of-range in token: "+tok);
    int base = (ab=='a') ? 0 : 16;
    return base + (r-1)*4 + (c-1); // 0..15 for A, 16..31 for B
}
static void load_placement_simple(const std::string& path,
                                  std::vector<std::pair<std::string,std::string>>& placement49)
{
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open placement file: "+path);
    placement49.clear(); placement49.reserve(49);
    std::string line; int lineno = 0;
    while (std::getline(fin, line)){
        lineno++;
        auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        auto notspace = [](int ch){ return !std::isspace(ch); };
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), notspace));
        line.erase(std::find_if(line.rbegin(), line.rend(), notspace).base(), line.end());
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string t1, t2;
        if (!(iss >> t1 >> t2))
            throw std::runtime_error("Need two tokens on line "+std::to_string(lineno));
        auto norm = [](const std::string& t){
            size_t p = t.find(':');
            std::string s = (p==std::string::npos) ? t : t.substr(p+1);
            char ab = std::tolower(s[0]);
            std::string rc = s.substr(1,2);
            std::string AB = (ab=='a') ? "A:a" : "B:b";
            return AB + rc;
        };
        placement49.push_back({norm(t1), norm(t2)});
    }
    if ((int)placement49.size() != 49){
        throw std::runtime_error("Parsed "+std::to_string(placement49.size())+
                                 " core lines; expected 49 (non-empty, non-comment).");
    }
}
static void build_owner_from_placement_simple(
    const std::vector<std::pair<std::string,std::string>>& placement49,
    int world)
{
    if (world < 49) throw std::runtime_error("Need 49 MPI ranks; got "+std::to_string(world));
    for (int id=0; id<32; ++id) OWNER[id] = id % world; // fallback (unused after pins)
    auto parse_tok = [](const std::string& tok)->int{ return sym_to_id_simple(tok); };
    for (int r=0; r<49; ++r){
        int id1 = parse_tok(placement49[r].first);
        int id2 = parse_tok(placement49[r].second);
        OWNER[id1] = r; OWNER[id2] = r;
    }
}

// -------------------- Demo block initialization --------------------
static inline M2 make_block(int seed){
    M2 m{};
    m.v[0][0] = seed+1; m.v[0][1] = 0.1*seed;
    m.v[1][0] = 0.1*seed; m.v[1][1] = seed+2;
    return m;
}

// -------------------- Depth-2 Strassen generator --------------------
using Lin = std::map<int,int>; // id -> coeff

static inline void lin_accum(Lin& dst, const Lin& src, int s){
    for (auto& kv: src) dst[kv.first] += s * kv.second;
}
static inline Lin singleton(int id){ return Lin{{id,1}}; }

struct Lin2x2 { Lin e[2][2]; };

static inline void build_quadrants_A(std::array<Lin2x2,4>& Qa){
    auto at = [&](int r,int c)->Lin{ return singleton(r*4 + c); }; // 0..15
    Qa[0].e[0][0]=at(0,0); Qa[0].e[0][1]=at(0,1); Qa[0].e[1][0]=at(1,0); Qa[0].e[1][1]=at(1,1);
    Qa[1].e[0][0]=at(0,2); Qa[1].e[0][1]=at(0,3); Qa[1].e[1][0]=at(1,2); Qa[1].e[1][1]=at(1,3);
    Qa[2].e[0][0]=at(2,0); Qa[2].e[0][1]=at(2,1); Qa[2].e[1][0]=at(3,0); Qa[2].e[1][1]=at(3,1);
    Qa[3].e[0][0]=at(2,2); Qa[3].e[0][1]=at(2,3); Qa[3].e[1][0]=at(3,2); Qa[3].e[1][1]=at(3,3);
}
static inline void build_quadrants_B(std::array<Lin2x2,4>& Qb){
    auto bt = [&](int r,int c)->Lin{ return singleton(16 + r*4 + c); }; // 16..31
    Qb[0].e[0][0]=bt(0,0); Qb[0].e[0][1]=bt(0,1); Qb[0].e[1][0]=bt(1,0); Qb[0].e[1][1]=bt(1,1);
    Qb[1].e[0][0]=bt(0,2); Qb[1].e[0][1]=bt(0,3); Qb[1].e[1][0]=bt(1,2); Qb[1].e[1][1]=bt(1,3);
    Qb[2].e[0][0]=bt(2,0); Qb[2].e[0][1]=bt(2,1); Qb[2].e[1][0]=bt(3,0); Qb[2].e[1][1]=bt(3,1);
    Qb[3].e[0][0]=bt(2,2); Qb[3].e[0][1]=bt(2,3); Qb[3].e[1][0]=bt(3,2); Qb[3].e[1][1]=bt(3,3);
}
static inline Lin2x2 combine_quads(const std::array<Lin2x2,4>& Q, const std::vector<std::pair<int,int>>& parts){
    Lin2x2 X;
    for (auto [qi,sg]: parts){
        for (int i=0;i<2;i++) for (int j=0;j<2;j++) lin_accum(X.e[i][j], Q[qi].e[i][j], sg);
    }
    return X;
}
struct SubLeaf { Lin LA, LB; std::vector<std::pair<int,int>> subC; }; // subC: (subpos, sign)
static inline void make_sub_leaves(const Lin2x2& X, const Lin2x2& Y, std::array<SubLeaf,7>& out){
    // Standard Strassen on 2x2
    out[0].LA = X.e[0][0]; lin_accum(out[0].LA, X.e[1][1], +1);
    out[0].LB = Y.e[0][0]; lin_accum(out[0].LB, Y.e[1][1], +1);
    out[0].subC = {{0,+1},{3,+1}};
    out[1].LA = X.e[1][0]; lin_accum(out[1].LA, X.e[1][1], +1);
    out[1].LB = Y.e[0][0];
    out[1].subC = {{2,+1},{3,-1}};
    out[2].LA = X.e[0][0];
    out[2].LB = Y.e[0][1]; lin_accum(out[2].LB, Y.e[1][1], -1);
    out[2].subC = {{1,+1},{3,+1}};
    out[3].LA = X.e[1][1];
    out[3].LB = Y.e[1][0]; lin_accum(out[3].LB, Y.e[0][0], -1);
    out[3].subC = {{0,+1},{2,+1}};
    out[4].LA = X.e[0][0]; lin_accum(out[4].LA, X.e[0][1], +1);
    out[4].LB = Y.e[1][1];
    out[4].subC = {{0,-1},{1,+1}};
    out[5].LA = X.e[1][0]; lin_accum(out[5].LA, X.e[0][0], -1);
    out[5].LB = Y.e[0][0]; lin_accum(out[5].LB, Y.e[0][1], +1);
    out[5].subC = {{3,+1}};
    out[6].LA = X.e[0][1]; lin_accum(out[6].LA, X.e[1][1], -1);
    out[6].LB = Y.e[1][0]; lin_accum(out[6].LB, Y.e[1][1], +1);
    out[6].subC = {{0,+1}};
}
// Top-level recombination mapping for each Mk to top-level C quadrants (0..3):
static inline const std::vector<std::pair<int,int>>& topC_for_k(int k){
    static const std::vector<std::pair<int,int>> tbl[7] = {
        /*M1*/ {{0,+1},{3,+1}},
        /*M2*/ {{2,+1},{3,-1}},
        /*M3*/ {{1,+1},{3,+1}},
        /*M4*/ {{0,+1},{2,+1}},
        /*M5*/ {{0,-1},{1,+1}},
        /*M6*/ {{3,+1}},
        /*M7*/ {{0,+1}},
    };
    return tbl[k];
}
static inline void top_op_for_k(int k,
                                std::vector<std::pair<int,int>>& Aparts,
                                std::vector<std::pair<int,int>>& Bparts)
{
    Aparts.clear(); Bparts.clear();
    switch(k){
        case 0: Aparts={{0,+1},{3,+1}}; Bparts={{0,+1},{3,+1}}; break;
        case 1: Aparts={{2,+1},{3,+1}}; Bparts={{0,+1}}; break;
        case 2: Aparts={{0,+1}};        Bparts={{1,+1},{3,-1}}; break;
        case 3: Aparts={{3,+1}};        Bparts={{2,+1},{0,-1}}; break;
        case 4: Aparts={{0,+1},{1,+1}}; Bparts={{3,+1}}; break;
        case 5: Aparts={{2,+1},{0,-1}}; Bparts={{0,+1},{1,+1}}; break;
        case 6: Aparts={{1,+1},{3,-1}}; Bparts={{2,+1},{3,+1}}; break;
    }
}
// NOTE: fixed typo in previous line; replace `{3;+1}` with `{3,+1}`
static inline int Cid_from_qs(int q, int s){
    int qr = (q/2), qc = (q%2);
    int sr = (s/2), sc = (s%2);
    int r = qr*2 + sr;
    int c = qc*2 + sc;
    return r*4 + c; // 0..15
}
static std::array<Leaf,49> build_depth2_recipes(){
    std::array<Leaf,49> Ls;
    std::array<Lin2x2,4> QA, QB;
    build_quadrants_A(QA); build_quadrants_B(QB);
    int idx = 0;
    for (int k=0;k<7;k++){
        std::vector<std::pair<int,int>> Aparts, Bparts;
        top_op_for_k(k, Aparts, Bparts);
        Lin2x2 X = combine_quads(QA, Aparts);
        Lin2x2 Y = combine_quads(QB, Bparts);
        std::array<SubLeaf,7> subs; make_sub_leaves(X, Y, subs);
        const auto& topC = topC_for_k(k);
        for (int m=0;m<7;m++){
            Leaf leaf;
            for (auto& kv: subs[m].LA) if (kv.second!=0)
                leaf.Aterms.push_back({ kv.first, (int8_t)(kv.second>=0?+1:-1) });
            for (auto& kv: subs[m].LB) if (kv.second!=0)
                leaf.Bterms.push_back({ kv.first, (int8_t)(kv.second>=0?+1:-1) });
            for (auto [q, tq]: topC)
                for (auto [s, ts]: subs[m].subC){
                    int cid = Cid_from_qs(q, s);
                    int sign = tq * ts;
                    leaf.Couts.push_back({ cid, (int8_t)(sign>=0?+1:-1) });
                }
            Ls[idx++] = std::move(leaf);
        }
    }
    return Ls; // 49 leaves
}

// -------------------- P2P fetch: Alltoallv requests + replies --------------------
static void p2p_fetch_many(const std::vector<int>& remote_ids,
                           const Store& store,
                           int me, int world,
                           std::unordered_map<int,M2>& out_cache)
{
    // Group requested ids by owner (exclude self)
    std::vector<int> send_counts(world, 0);
    std::vector<std::vector<int>> by_owner(world);
    for (int id : remote_ids){
        int ow = OWNER[id];
        if (ow == me) continue;
        by_owner[ow].push_back(id);
    }
    for (int r=0; r<world; ++r) send_counts[r] = (int)by_owner[r].size();

    // Exchange request counts so owners know how many IDs to expect from each rank
    std::vector<int> recv_counts(world, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Build send/recv buffers for ID exchange
    auto prefix = [](const std::vector<int>& v){
        std::vector<int> p(v.size()+1,0);
        for (size_t i=0;i<v.size();++i) p[i+1]=p[i]+v[i];
        return p;
    };
    // Flatten ids to send
    std::vector<int> sdispls(world,0), rdispls(world,0);
    std::vector<int> send_ids;
    {
        int total_send = 0;
        for (int r=0;r<world;++r) total_send += send_counts[r];
        send_ids.reserve(total_send);
        for (int r=0;r<world;++r){
            sdispls[r] = (int)send_ids.size();
            send_ids.insert(send_ids.end(), by_owner[r].begin(), by_owner[r].end());
        }
        // recv displs
        int acc=0;
        for (int r=0;r<world;++r){ rdispls[r]=acc; acc += recv_counts[r]; }
    }
    int total_recv_ids = 0;
    for (int r=0;r<world;++r) total_recv_ids += recv_counts[r];
    std::vector<int> recv_ids(total_recv_ids, -1);

    // Exchange ID lists: requesters -> owners
    MPI_Alltoallv(send_ids.data(), send_counts.data(), sdispls.data(), MPI_INT,
                  recv_ids.data(), recv_counts.data(), rdispls.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Owners prepare payloads (4 doubles per requested id, in-place packed in same order)
    std::vector<int> send_counts_d(world, 0), sdispls_d(world, 0);
    std::vector<int> recv_counts_d(world, 0), rdispls_d(world, 0);

    // For doubles: counts are 4 * (#ids)
    for (int r=0;r<world;++r){
        // I (as owner) will send to rank r as many doubles as they requested from me
        send_counts_d[r] = 4 * recv_counts[r];
        // I (as requester) will receive from rank r as many doubles as I asked them for
        recv_counts_d[r] = 4 * send_counts[r];
    }
    // Build displacements for doubles
    {
        int acc=0; for (int r=0;r<world;++r){ sdispls_d[r]=acc; acc += send_counts_d[r]; }
        acc=0; for (int r=0;r<world;++r){ rdispls_d[r]=acc; acc += recv_counts_d[r]; }
    }

    // Pack send (owner side)
    std::vector<double> send_data_d( std::accumulate(send_counts_d.begin(), send_counts_d.end(), 0), 0.0 );
    for (int src=0; src<world; ++src){
        int n_ids = recv_counts[src];
        if (n_ids==0) continue;
        int off_id = rdispls[src];             // where that src's ID segment begins in recv_ids
        int off_db = sdispls_d[src];           // where to pack doubles for that src
        for (int i=0;i<n_ids;++i){
            int id = recv_ids[off_id + i];
            auto it = store.data.find(id);
            if (it==store.data.end()){
                // Should not happen if OWNER table is consistent; send zeros defensively.
                double* p = &send_data_d[off_db + 4*i];
                p[0]=p[1]=p[2]=p[3]=0.0;
            }else{
                pack_M2(it->second, &send_data_d[off_db + 4*i]);
            }
        }
    }

    // Receive buffer for requested data (requester side)
    std::vector<double> recv_data_d( std::accumulate(recv_counts_d.begin(), recv_counts_d.end(), 0), 0.0 );

    // Owners -> Requesters: send packed 2x2 blocks (as 4 doubles per id)
    MPI_Alltoallv(send_data_d.data(), send_counts_d.data(), sdispls_d.data(), MPI_DOUBLE,
                  recv_data_d.data(), recv_counts_d.data(), rdispls_d.data(), MPI_DOUBLE,
                  MPI_COMM_WORLD);

    // Unpack on requester side back into cache, following *the same ID order we sent*
    for (int dst=0; dst<world; ++dst){
        int n_ids = send_counts[dst];
        if (n_ids==0) continue;
        int off_id = sdispls[dst];        // the IDs we sent to that owner
        int off_db = rdispls_d[dst];      // the doubles we received from that owner
        for (int i=0;i<n_ids;++i){
            int id = send_ids[off_id + i];
            const double* p = &recv_data_d[off_db + 4*i];
            out_cache[id] = unpack_M2(p);
        }
    }
}

// -------------------- Main --------------------
int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int me=0, world=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    if (world < 49) {
        if (me==0) std::fprintf(stderr, "ERROR: need 49 ranks; got %d\n", world);
        MPI_Finalize();
        return 1;
    }
    if (argc < 2){
        if (me==0) std::fprintf(stderr, "Usage: %s <placement_file>\n", argv[0]);
        MPI_Finalize();
        return 2;
    }

    double t0_all = MPI_Wtime();
    double t_fetch = 0.0, t_compute = 0.0;
    unsigned long long approx_bytes_recv = 0ULL; // 4 doubles per fetched block

    // Load placement & build OWNER
    std::vector<std::pair<std::string,std::string>> placement;
    try {
        load_placement_simple(argv[1], placement);
        build_owner_from_placement_simple(placement, world);
    } catch (const std::exception& e){
        if (me==0) std::fprintf(stderr, "Placement error: %s\n", e.what());
        MPI_Finalize();
        return 3;
    }

    // Local store: only blocks owned by this rank
    Store store;
    for (int id=0; id<32; ++id)
        if (owner_of(id, world) == me)
            store.data[id] = make_block(id+1);

    // Build Strassen depth-2 leaves
    auto recipes = build_depth2_recipes();

    // Execute leaf (ranks 0..48)
    std::array<double,64> sendbuf{}; // 16 blocks * 4 doubles
    std::array<double,64> recvbuf{};
    sendbuf.fill(0.0); recvbuf.fill(0.0);

    if (me < 49){
        const Leaf& leaf = recipes[me];

        // Collect unique needed A/B ids
        std::vector<int> need_ids;
        need_ids.reserve(leaf.Aterms.size()+leaf.Bterms.size());
        for (auto t: leaf.Aterms) need_ids.push_back(t.id);
        for (auto t: leaf.Bterms) need_ids.push_back(t.id);
        std::sort(need_ids.begin(), need_ids.end());
        need_ids.erase(std::unique(need_ids.begin(), need_ids.end()), need_ids.end());

        // Determine which are remote (not in local store)
        std::vector<int> to_fetch;
        to_fetch.reserve(need_ids.size());
        for (int id: need_ids) if (!store.data.count(id)) to_fetch.push_back(id);

        std::unordered_map<int,M2> cache; cache.reserve(to_fetch.size()+4);

        double t0 = MPI_Wtime();
        if (!to_fetch.empty()){
            // P2P two-phase: send ID lists to owners, receive payloads back.
            p2p_fetch_many(to_fetch, store, me, world, cache);
            approx_bytes_recv += static_cast<unsigned long long>(to_fetch.size()) * sizeof(double) * 4ULL;
        }
        double t1 = MPI_Wtime(); t_fetch += (t1 - t0);

        // Helper to retrieve block (local or from cache)
        auto getblk = [&](int id)->const M2& {
            if (auto it=store.data.find(id); it!=store.data.end()) return it->second;
            return cache.at(id);
        };

        // Evaluate linear forms and compute the single 2x2 product
        auto combine = [&](const std::vector<Term>& ts)->M2{
            M2 acc{}; // zero
            for (auto t: ts){
                const M2& b = getblk(t.id);
                if (t.s==+1) acc = add2(acc, b, 1.0, +1.0);
                else         acc = add2(acc, b, 1.0, -1.0);
            }
            return acc;
        };

        double t2 = MPI_Wtime();
        M2 Atil = combine(leaf.Aterms);
        M2 Btil = combine(leaf.Bterms);
        M2 P    = mul2(Atil, Btil);
        double t3 = MPI_Wtime(); t_compute += (t3 - t2);

        // Build contributions to C (only touched blocks)
        std::array<M2,16> Ccontrib{};
        for (auto t: leaf.Couts){
            if (t.s==+1) Ccontrib[t.id] = add2(Ccontrib[t.id], P, 1.0, +1.0);
            else         Ccontrib[t.id] = add2(Ccontrib[t.id], P, 1.0, -1.0);
        }
        for (int k=0; k<16; ++k) pack_M2(Ccontrib[k], &sendbuf[4*k]);
    }

    // Reduce to rank 0
    MPI_Reduce(sendbuf.data(), recvbuf.data(), 64, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Metrics
    double t_rank_total = MPI_Wtime() - t0_all;
    double sum_fetch=0, sum_compute=0, sum_total=0, max_total=0;
    unsigned long long total_bytes_recv=0ULL;

    MPI_Reduce(&t_fetch,   &sum_fetch,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_compute, &sum_compute, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_rank_total, &sum_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_rank_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&approx_bytes_recv, &total_bytes_recv, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (me==0){
        std::array<M2,16> Cblocks{};
        for (int k=0;k<16;k++) Cblocks[k] = unpack_M2(&recvbuf[4*k]);
        double C[8][8]{}; stitch8(C, Cblocks);

        double avg_fetch   = sum_fetch   / world;
        double avg_compute = sum_compute / world;
        double avg_total   = sum_total   / world;

        std::ofstream fout("result.txt");
        fout.setf(std::ios::fixed); fout.precision(6);

        fout << "Result C (8x8 numeric matrix) — depth-2 Strassen (49 leaves, P2P two-sided)\n";
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                fout << C[r][c] << (c + 1 < 8 ? ' ' : '\n');
            }
        }
        fout << "\n--- Metrics ---\n";
        fout << "Average fetch time (s): " << avg_fetch << "\n";
        fout << "Average compute time (s): " << avg_compute << "\n";
        fout << "Average total time per rank (s): " << avg_total << "\n";
        fout << "Total job time (max over ranks, s): " << max_total << "\n";
        fout << "Approx payload bytes RECEIVED: " << total_bytes_recv
             << " (" << (total_bytes_recv/1024.0) << " KB)\n";
        fout << "World size: " << world << " (intended 49)\n";
        fout.close();

        std::printf("Result and metrics written to result.txt\n");
    }

    MPI_Finalize();
    return 0;
}
