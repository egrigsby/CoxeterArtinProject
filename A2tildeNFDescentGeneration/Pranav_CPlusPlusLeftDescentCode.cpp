#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
using namespace std;

string generateRandomString(int length)
{
    // Define the list of possible characters
    const string CHARACTERS
        = "abc";

    // Create a random number generator
    random_device rd;
    mt19937 generator(rd());

    // Create a distribution to uniformly select from all
    // characters
    uniform_int_distribution<> distribution(
        0, CHARACTERS.size() - 1);

    // Generate the random string
    string random_string;
    for (int i = 0; i < length; ++i) {
        random_string
            += CHARACTERS[distribution(generator)];
    }

    return random_string;
}

class Timer
{
private:
	// Type aliases to make accessing nested type easier
	using Clock = std::chrono::steady_clock;
	using Second = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<Clock> m_beg { Clock::now() };

public:
	void reset()
	{
		m_beg = Clock::now();
	}

	double elapsed() const
	{
		return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
	}
};

int order(char s){
    if (s=='a'){
        return 0;
    }
    if (s=='b'){
        return 1;
    }
    if (s=='c'){
        return 2;
    }
    if (s== 'd'){
        return 3;
    }
    if (s== 'e'){
        return 4;
    }
    if (s== 'f'){
        return 5;
    }
    else{
        return 6;
    }
}

char RootRefTable(char s, char a){
    char Table[][3] = {{'-', 'd', 'f'}, {'d', '-', 'e'}, {'f', 'e', '-'}, {'b', 'a', '+'}, {'+', 'c', 'b'}, {'c', '+', 'a'}};

    return Table[order(a)][order(s)];
}

string InsertChar(char t, string w, int k){
    if (k==0){
        return t + w;
    }
    else{
        return w.substr(0, k) + t + w.substr(k);
    }
}

/*string RemoveChar(string w, int k){
    if (k == 0){
        return w.substr(1);
    }
    else{
        return w.substr(0, k) + w.substr(k+1);
    }

} use .erase(k, k+1) instead*/

string Mult(char s, string w){
    char t = s, lambda = s;
    int k = 0;
    for(int i= 0; i<w.size(); i++){
        lambda = RootRefTable(w[i],lambda);
        if(lambda == '-'){
            return w.erase(k,k+1);
        }
        else if(lambda=='+'){
            return InsertChar(t, w, k);
        }
        else if(order(lambda)<order(w[i])){
            k = i+1;
            t = lambda;
        }
    }
    return InsertChar(t, w, k);
}

bool isLeftDescent(char s, string w){
    char t = s, lambda = s;
    int k = 0;
    for(int i= 0; i<w.size(); i++){
        lambda = RootRefTable(w[i],lambda);
        if(lambda == '-'){
            return true;
        }
        else if(lambda=='+'){
            return false;
        }
    }
    return false;
}

struct PathData{
    vector<string> elementList;
    vector<string> descentsList;
};

struct PathData DescentPath(string w){
    struct PathData returningPathData;
    if (w.size()==1){
        returningPathData.elementList.push_back(w);
        returningPathData.descentsList.push_back(w);
        return returningPathData;
    }
    int wordSize = w.size();
    string x = w.substr(wordSize-1);
    returningPathData.elementList.push_back(w.substr(wordSize-1));
    string gens[3] = {"a", "b", "c"};
    for (int i = 1; i < wordSize; i++){
        string descent = "";
        string newx;
        //cout << i << " " << wordSize-i << endl;
        //cout << w.substr(wordSize-i-1, 1) << endl;
        for(int j =0; j<3; j++){
            if (!gens[j].compare(w.substr(wordSize-i-1, 1))){
                newx = Mult(gens[j][0],x);
                //cout << newx << endl;
                if(newx.size()<x.size()){
                    descent.append(gens[j]);
                }
                returningPathData.elementList.push_back(newx);
            }
            else{
                bool ifLD = isLeftDescent(gens[j][0], x);
                //cout << x + ", " + gens[j] << ": " << ifLD << endl; 
                if (ifLD){
                    descent.append(gens[j]);
                }

            }
            
        }
        x = newx;
        returningPathData.descentsList.push_back(descent);
    }
    string finaldescent = "";
    for(int j =0; j<3; j++){
            if (isLeftDescent(gens[j][0], x)){
                finaldescent.append(gens[j]);
            }
            
    }
    returningPathData.descentsList.push_back(finaldescent);
    return returningPathData;
}

string VectorValuesAsString(vector<string> x){
    string output = "[";
    for(int i =0; i<x.size()-1; i++){
        output.append(x[i] + ", ");
    }
    output.append(x[x.size()-1] + "]");
    return output;
}

int main () {
  int instances = 100000;
  int length = 200;
  Timer t;
  for(int i = 0; i< instances; i++){
    if (i % 1000 == 0){
        cout << i << endl;
    }
    string s = generateRandomString(length);
    struct PathData x = DescentPath(s);
  }
  cout << "It took " << t.elapsed() << " seconds at " << instances << " instances of length " << length << endl;
  return 0;
}