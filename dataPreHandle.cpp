#include<bits/stdc++.h>
using namespace std;
#define For(i,a,b) for(int i=(a);i<=(b);i++)
#define Rof(i,a,b) for(int i=(a);i>=(b);i--)
#define ll long long
#define wln putchar('\n')
template<class T1,class T2> void chkmin(T1 &x,T2 y){if(y<x)x=y;}
template<class T1,class T2> void chkmax(T1 &x,T2 y){if(y>x)x=y;}

#define MAXN 10004

int N = 7284;

string match_id,player1,player2,elapsed_time;
string set_no,game_no,point_no;
string p1_sets,p2_sets,p1_games,p2_games,p1_score,p2_score;
string server,serve_no,point_victor,p1_points_won,p2_points_won;
string game_victor,set_victor;
string p1_ace,p2_ace,p1_winner,p2_winner;
string winner_shot_type;
string p1_double_fault,p2_double_fault;
string p1_unf_err,p2_unf_err;
string p1_net_pt,p2_net_pt,p1_net_pt_won,p2_net_pt_won;
string p1_break_pt,p2_break_pt,p1_break_pt_won,p2_break_pt_won,p1_break_pt_missed,p2_break_pt_missed;
string p1_distance_run,p2_distance_run,rally_count,speed_mph,serve_width,serve_depth,return_depth;

struct node {
    string match_id,player1_first,player1_last,player2_first,player2_last,elapsed_time;
    int set_no,game_no,point_no;
    int p1_sets,p2_sets,p1_games,p2_games,p1_score,p2_score;
    string server,serve_no,point_victor,p1_points_won,p2_points_won;
    string game_victor,set_victor;
    string p1_ace,p2_ace,p1_winner,p2_winner;
    string winner_shot_type;
    string p1_double_fault,p2_double_fault;
    string p1_unf_err,p2_unf_err;
    string p1_net_pt,p2_net_pt,p1_net_pt_won,p2_net_pt_won;
    string p1_break_pt,p2_break_pt,p1_break_pt_won,p2_break_pt_won,p1_break_pt_missed,p2_break_pt_missed;
    string p1_distance_run,p2_distance_run,rally_count,speed_mph,serve_width,serve_depth,return_depth;
    int p1_side, p2_side;
    int winner;
    int serve;
}a[MAXN];

typedef pair<string, string> pss;

map<string, pss> mp;

int main()
{
    freopen("./assets/doc/output.csv", "r", stdin);
    // freopen("input.csv", "w", stdout);
    int n = 0;
    For(i, 1, N) {
        cin>>a[i].match_id>>a[i].player1_first>>a[i].player1_last>>a[i].player2_first>>a[i].player2_last>>a[i].elapsed_time>>a[i].set_no>>a[i].game_no>>a[i].point_no>>a[i].p1_sets>>a[i].p2_sets>>a[i].p1_games>>a[i].p2_games>>a[i].p1_score>>a[i].p2_score>>a[i].server>>a[i].serve_no>>a[i].point_victor>>a[i].p1_points_won>>a[i].p2_points_won>>a[i].game_victor>>a[i].set_victor>>a[i].p1_ace>>a[i].p2_ace>>a[i].p1_winner>>a[i].p2_winner>>a[i].winner_shot_type>>a[i].p1_double_fault>>a[i].p2_double_fault>>a[i].p1_unf_err>>a[i].p2_unf_err>>a[i].p1_net_pt>>a[i].p2_net_pt>>a[i].p1_net_pt_won>>a[i].p2_net_pt_won>>a[i].p1_break_pt>>a[i].p2_break_pt>>a[i].p1_break_pt_won>>a[i].p2_break_pt_won>>a[i].p1_break_pt_missed>>a[i].p2_break_pt_missed>>a[i].p1_distance_run>>a[i].p2_distance_run>>a[i].rally_count>>a[i].speed_mph>>a[i].serve_width>>a[i].serve_depth>>a[i].return_depth;
        if(mp.find(a[i].match_id) == mp.end())
            mp[a[i].match_id] = {a[i].player1_first+" "+a[i].player1_last, a[i].player2_first+" "+a[i].player2_last};
		continue;
        // cout << "in1"<<endl; 
		// cout<<a[i].match_id<<","<<a[i].player1_first + " "<<a[i].player1_last<<","<<a[i].player2_first + " "<<a[i].player2_last<<","<<a[i].elapsed_time<<","<<a[i].set_no<<","<<a[i].game_no<<","<<a[i].point_no<<","<<a[i].p1_sets<<","<<a[i].p2_sets<<","<<a[i].p1_games<<","<<a[i].p2_games<<","<<a[i].p1_score<<","<<a[i].p2_score<<","<<a[i].server<<","<<a[i].serve_no<<","<<a[i].point_victor<<","<<a[i].p1_points_won<<","<<a[i].p2_points_won<<","<<a[i].game_victor<<","<<a[i].set_victor<<","<<a[i].p1_ace<<","<<a[i].p2_ace<<","<<a[i].p1_winner<<","<<a[i].p2_winner<<","<<a[i].winner_shot_type<<","<<a[i].p1_double_fault<<","<<a[i].p2_double_fault<<","<<a[i].p1_unf_err<<","<<a[i].p2_unf_err<<","<<a[i].p1_net_pt<<","<<a[i].p2_net_pt<<","<<a[i].p1_net_pt_won<<","<<a[i].p2_net_pt_won<<","<<a[i].p1_break_pt<<","<<a[i].p2_break_pt<<","<<a[i].p1_break_pt_won<<","<<a[i].p2_break_pt_won<<","<<a[i].p1_break_pt_missed<<","<<a[i].p2_break_pt_missed<<","<<a[i].p1_distance_run<<","<<a[i].p2_distance_run<<","<<a[i].rally_count<<","<<a[i].speed_mph<<","<<a[i].serve_width<<","<<a[i].serve_depth<<","<<a[i].return_depth<<endl;
        // cout << "in2" << endl;
        // cout << "what: " << a[i].serve_depth << " :what" << endl;
		if(a[i].game_no == 1) {
            a[i].p1_side = 1, a[i].p2_side = 2;
        }
        else if(a[i].game_no == a[i - 1].game_no) {
            a[i].p1_side = a[i - 1].p1_side, a[i].p2_side = a[i - 1].p2_side;
        }
        else {
            int x = a[i].game_no;
            a[i].p1_side = a[i - 1].p1_side, a[i].p2_side = a[i - 1].p2_side;
            if(x % 2 == 0) swap(a[i].p1_side, a[i].p2_side);
        }
        if(a[i].p1_score < 40 && a[i].p2_score < 40) {
            a[i].p1_score /= 15, a[i].p2_score /= 15;
        }
        else if(a[i].p1_score < 40 && a[i].p2_score == 40) {
            a[i].p1_score /= 15, a[i].p2_score = 3;
        }
        else if(a[i].p2_score < 40 && a[i].p1_score == 40) {
            a[i].p2_score /= 15, a[i].p1_score = 3;
        }
        if(a[i].p1_score < 40 && a[i].p2_score < 40) {
            a[i].p1_score /= 15, a[i].p2_score /= 15;
        }
        else if(a[i].p1_score < 40 && a[i].p2_score == 40) {
            a[i].p1_score /= 15, a[i].p2_score = 3;
        }
        else if(a[i].p2_score < 40 && a[i].p1_score == 40) {
            a[i].p2_score /= 15, a[i].p1_score = 3;
        }
        else {
            if(a[i].p1_score == 50) a[i].p1_score = a[i - 1].p1_score + 1, a[i].p2_score = a[i - 1].p2_score;
            else if(a[i].p2_score == 50) a[i].p2_score = a[i - 1].p2_score + 1, a[i].p1_score = a[i - 1].p1_score;
            else if(a[i - 1].p1_score < 3 || a[i - 1].p2_score < 3) a[i].p1_score = a[i].p2_score = 3;
            else {
                if(a[i - 1].p1_score > a[i - 1].p2_score) a[i].p2_score = a[i - 1].p2_score + 1, a[i].p1_score = a[i - 1].p1_score;
                else a[i].p1_score = a[i - 1].p1_score + 1, a[i].p2_score = a[i - 1].p2_score;
            }
        }
		// cout<<a[i].p1_score<<","<<a[i].p2_score<<endl;
    }
    For(i, 1, n) {
        if(i == 1 || a[i].p1_score < a[i - 1].p1_score || a[i].p2_score < a[i - 1].p2_score) {
            if(a[i].p1_score) a[i].winner = 1;
            else a[i].winner = 2;
        }
        else {
            if(a[i].p1_score > a[i - 1].p1_score) a[i].winner = 1;
            else a[i].winner = 2;
        }
    }
    int x = 0, y = 0;
    cout << "p1_points_won" << "," << "p2_points_won" << "," << "winner" << "," << "server" << endl;
    For(i, 1, n) {
        if(a[i].winner == 1) x++;
        else y++;
        cout << x << "," << y << "," << a[i].winner << "," << a[i].serve << "\n";
        // cout << a[i].p1_score << "," << a[i].p2_score << "," << a[i].winner << endl;
    }
}
