*************************************************************************/
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<ctime>
#include <math.h> 
using namespace std;

const int MAXN = 50;       // Max neurons in any layer
const int MAXPATS = 5000;  // Max training patterns

// mlp paramaters
long  NumIts ;    // Max training iterations
int   NumHN  ;    // Number of hidden layers
int   NumHN1 ;    // Number of neurons in hidden layer 1
int   NumHN2 ;    // Number of neurons in hidden layer 2
int   NumHN3 ;    // Number of neurons in hidden layer 3
int   NumHN4 ;    // Number of neurons in hidden layer 4
float LrnRate;    // Learning rate 
float Mtm1   ;    // Momentum(t-1) 
float Mtm2   ;    // Momentum(t-2) 
float ObjErr ;    // Objective error 

// mlp weights
float **w1,**w11,**w111;// 1st layer wts
float **w2,**w22,**w222;// 2nd layer wts
float **w3,**w33,**w333;// 3rd layer wts
float **w4,**w44,**w444;// 4rth layer wts
float **w5,**w55,**w555;// 5th layer wts


void TrainNet2(float **x,float **d,int NumIPs,int NumOPs,int NumPats, int Ordering);
void TrainNet3(float **x,float **d,int NumIPs,int NumOPs,int NumPats);
void TrainNet4(float **x,float **d,int NumIPs,int NumOPs,int NumPats);
void TrainNet5(float **x,float **d,int NumIPs,int NumOPs,int NumPats);

void TestNet2(float **x,float **d,int NumIPs,int NumOPs,int NumPats);
void TestNet3(float **x,float **d,int NumIPs,int NumOPs,int NumPats);

float **Aloc2DAry(int m,int n);
void Free2DAry(float **Ary2D,int n);
long random_gen(long max);

int main(){
  ifstream fin;
  int i,j,NumIPs,NumOPs,NumTrnPats,NumTstPats,Ordering;   
  
  char Line[500],Tmp[20],FName[20];
  
  cout<<"Enter data filename: ";
  cin>>FName; cin.ignore();
  fin.open(FName);
  if(!fin.good()){cout<<"File not found!\n";exit(1);}
  

  //read data specs...
  do{fin.getline(Line,500);}while(Line[0]==';'); //eat comments
  sscanf(Line,"%s%d",Tmp,&NumIPs); 
  fin>>Tmp>>NumOPs;     //Number of outputs 
  fin>>Tmp>>NumTrnPats; //Total number of training patterns 
  fin>>Tmp>>NumTstPats; //Total testing patterns
  fin>>Tmp>>NumIts;  //training iterations
  fin>>Tmp>>NumHN;  //no. of hidden layers
  
  i=NumHN;
  
  //if more than one layer network
  if(i-- > 0)fin>>Tmp>>NumHN1;
  if(i-- > 0)fin>>Tmp>>NumHN2;
  if(i-- > 0)fin>>Tmp>>NumHN3;
  if(i-- > 0)fin>>Tmp>>NumHN4;
  
  ///read learn_rate,momentum, objective ...
  fin>>Tmp>>LrnRate;
  fin>>Tmp>>Mtm1;
  fin>>Tmp>>Mtm2;
  fin>>Tmp>>ObjErr;
  fin>>Tmp>>Ordering;

  //general error checking..
    
  if( NumIPs<1||NumIPs>MAXN||NumOPs<1||NumOPs>MAXN||
		NumTrnPats<1||NumTrnPats>MAXPATS||NumTrnPats<1||NumTrnPats>MAXPATS||
      NumIts<1||NumIts>20e6||NumHN1<0||NumHN1>900||
      LrnRate<0||LrnRate>1||Mtm1<0||Mtm1>10||Mtm2<0||Mtm2>10||ObjErr<0||ObjErr>10
    ){ cout<<"Invalid specs in data file!\n"; exit(1); }

  float **IPTrnData = Aloc2DAry(NumTrnPats,NumIPs);
  float **OPTrnData = Aloc2DAry(NumTrnPats,NumOPs);
  float **IPTstData = Aloc2DAry(NumTstPats,NumIPs);
  float **OPTstData = Aloc2DAry(NumTstPats,NumOPs);
  
  //simultaneously read data into the two 2d arrays IPTrnData,OPTrnData of length 'NumTrnPats x NumIPs' and 'NumTrnPats x NumOPs' respectively
  
  for(i=0;i<NumTrnPats;i++){
	 for(j=0;j<NumIPs;j++)
		fin>>IPTrnData[i][j];
	 for(j=0;j<NumOPs;j++)
		fin>>OPTrnData[i][j];
  }
    
    //simultaneously read data into the two 2d arrays IPTrnData,OPTrnData of length 'NumTrnPats x NumIPs' and 'NumTrnPats x NumOPs' respectively
  for(i=0;i<NumTstPats;i++){
	 for(j=0;j<NumIPs;j++)
		fin>>IPTstData[i][j];
	 for(j=0;j<NumOPs;j++)
		fin>>OPTstData[i][j];
  }
  fin.close();
  
  //---------------------------file reading ends here ---------------------------
    
  //call the appropriate TestNet function  
  switch(NumHN){
    case 1:
      TrainNet2(IPTrnData,OPTrnData,NumIPs,NumOPs,NumTrnPats,Ordering);
      TestNet2(IPTstData,OPTstData,NumIPs,NumOPs,NumTstPats);
      break;
    case 2:
       cout<<"Running TrainNet3: IP:"<<NumIPs<<"H1:"<<NumHN1<<"H2:"<<NumHN2<<"OP:"<<NumOPs<<endl;
       TrainNet3(IPTrnData,OPTrnData,NumIPs,NumOPs,NumTrnPats);  //3 layer 
       TestNet3(IPTstData,OPTstData,NumIPs,NumOPs,NumTstPats);
      break;
    case 3:
       TrainNet4(IPTrnData,OPTrnData,NumIPs,NumOPs,NumTrnPats); //4 layer
      break;
    case 4:
       TrainNet5(IPTrnData,OPTrnData,NumIPs,NumOPs,NumTrnPats); //5 layer
      break;
   }
  
  //use the test data to test the network 
  
    
  Free2DAry(IPTrnData,NumTrnPats);
  Free2DAry(OPTrnData,NumTrnPats);
  Free2DAry(IPTstData,NumTstPats);
  Free2DAry(OPTstData,NumTstPats);
  cout<<"End of program.\n";
  system("PAUSE");
  return 0;
}

void TrainNet2 (float **x,float **d,int NumIPs,int NumOPs,int NumPats,int Ordering){
// Trains 2 layer back propagation neural network
// x[][]=>input data, d[][]=>desired output data
  int input_index;
  float *h1 = new float[NumHN1]; // O/Ps of hidden layer

  float *y  = new float[NumOPs]; // O/P of Net

  float *ad1= new float[NumHN1]; // HN1 back prop errors
  float *ad2= new float[NumOPs]; // O/P back prop errors

  float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
  int p,i,j;     // for loops indexes
  long ItCnt=0;  // Iteration counter
  long NumErr=0; // Error counter (added for spiral problem)

  ofstream myfile;
  myfile.open ("experiment1.csv");
  myfile << "Train results.\n";
  myfile << "IternationCount,MinErr,AverageErr,MaxError,PercentageError,PatternError,\n";
     

  cout<<"TrainNet2: IP:"<<NumIPs<<" H1:"<<NumHN1<<" OP:"<<NumOPs<<endl;

  // Allocate memory for weights
  w1   = Aloc2DAry(NumIPs,NumHN1);// 1st layer wts
  w11  = Aloc2DAry(NumIPs,NumHN1);
  w111 = Aloc2DAry(NumIPs,NumHN1);

  w2   = Aloc2DAry(NumHN1,NumOPs);// 2nd layer wts
  w22  = Aloc2DAry(NumHN1,NumOPs);
  w222 = Aloc2DAry(NumHN1,NumOPs);
    
    
  // Init wts between -0.5 and +0.5
  srand(time(0));
  for(i=0;i<NumIPs;i++)
    for(j=0;j<NumHN1;j++)
    w1[i][j]=w11[i][j]=w111[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN1;i++)
    for(j=0;j<NumOPs;j++)
      w2[i][j]=w22[i][j]=w222[i][j]= float(rand())/RAND_MAX - 0.5;

    
  for(;;){// Main learning loop
    MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;


    for(p=0;p<NumPats;p++){ // for each pattern...

      //Selectable ordering 
      int rand_index = random_gen(NumPats-1);

      if (Ordering == 0){
        input_index = p;
      }
      //random selection of data 
      else if (Ordering == 1){
        input_index = rand_index ;
      }

      // Cal neural network output
      for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
        float in=0;
        for(j=0;j<NumIPs;j++)
          in+=w1[j][i]*x[input_index][j];
        h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
      }
      for(i=0;i<NumOPs;i++){ // Cal O/P of output layer
        float in=0;
        for(j=0;j<NumHN1;j++){
          in+=w2[j][i]*h1[j];
        }
        //y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        y[i] = tanh(in);

      }
      // Cal error for this pattern
      PatErr=0.0;
      for(i=0;i<NumOPs;i++){
        float err=y[i]-d[p][i]; // actual-desired O/P
        if(err>0)PatErr+=err; else PatErr-=err;
        NumErr += ((y[i]<0.5&&d[input_index][i]>=0.5)||(y[i]>=0.5 && d[input_index][i]<0.5));//added for binary classification problem

        //cout << "desiredouput:"<< d[p][i]<<"output:" << y[i] << "isError" << ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5)) << "NumErr"<< NumErr<<endl;

      }
      if(PatErr<MinErr)MinErr=PatErr;
      if(PatErr>MaxErr)MaxErr=PatErr;
      AveErr+=PatErr;
        
      // Learn pattern with back propagation
      for(i=0;i<NumOPs;i++){ // error correction layer 2
        //ad2[i]=(d[p][i]-y[i])*y[i]*(1.0-y[i]);
        ad2[i]=(d[input_index][i]-y[i])*(1-tanh(y[i]))*(1+tanh(y[i]));
      }

      for(i=0;i<NumHN1;i++){ //error correction layer 1
        float err=0.0;
        for(j=0;j<NumOPs;j++)
          err+=ad2[j]*w2[i][j];
        ad1[i]=err*h1[i]*(1.0-h1[i]);
      }

      for(i=0;i<NumOPs;i++){ // Modify layer 2 wts
        
        for(j=0;j<NumHN1;j++){
          w2[j][i]+=LrnRate*h1[j]*ad2[i]+
                    Mtm1*(w2[j][i]-w22[j][i])+
                    Mtm2*(w22[j][i]-w222[j][i]);
          w222[j][i]=w22[j][i];
          w22[j][i]=w2[j][i];
        }
      }
      for(i=0;i<NumHN1;i++){ // Modify layer 1 wts
        
        for(j=0;j<NumIPs;j++){
          w1[j][i]+=LrnRate * x[input_index][j]*ad1[i]+
                    Mtm1*(w1[j][i]-w11[j][i])+
                    Mtm2*(w11[j][i]-w111[j][i]);
          w111[j][i]=w11[j][i];
          w11[j][i]=w1[j][i];
        }
      }
    }// end for each pattern
     
    ItCnt++;
    AveErr/=NumPats;
    float PcntErr = NumErr/float(NumPats) * 100.0;
    cout.setf(ios::fixed|ios::showpoint);
    
    cout<<setprecision(6)<<setw(6)<<ItCnt<<": "<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<<"pcterr"<<PcntErr<< setw(12) <<"patterror" << PatErr<<endl;
      
    myfile << ItCnt <<"," << MinErr << "," << AveErr << "," << MaxErr  <<"," <<  PcntErr << "," << PatErr <<"\n";

    if((AveErr<=ObjErr)||(ItCnt==NumIts)) break;

  }// end main learning loop

  myfile.close();
  // Free memory
  delete h1; delete y; 
  delete ad1; delete ad2;
}


// Assumes 0 <= max <= RAND_MAX
// Returns in the half-open interval [0, max]
long random_gen(long max) {
  unsigned long
    // max <= RAND_MAX < ULONG_MAX, so this is okay.
    num_bins = (unsigned long) max + 1,
    num_rand = (unsigned long) RAND_MAX + 1,
    bin_size = num_rand / num_bins,
    defect   = num_rand % num_bins;

  long x;
  do {
   x = random();
  }
  // This is carefully written not to overflow
  while (num_rand - defect <= (unsigned long)x);

  // Truncated division is intentional
  return x/bin_size;
}

//-----------------------------------------------------
//TestNet: Tests 2 layer back propagation neural network
//------------------------------------------------------

void TestNet2(float **x,float **d,int NumIPs,int NumOPs,int NumPats){
  
  //x[][]=>input data, d[][]=>desired output data

  float *h1 = new float[NumHN1]; // O/Ps of hidden layer

  float *y  = new float[NumOPs]; // O/P of Net

  float *ad1= new float[NumHN1]; // HN1 back prop errors
  float *ad2= new float[NumOPs]; // O/P back prop errors

  float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
  int p,i,j;     // for loops indexes
  long ItCnt=0;  // Iteration counter
  long NumErr=0; // Error counter (added for spiral problem)

  cout<<"TestNet: IP:"<<NumIPs<<" H1:"<<NumHN1<<" OP:"<<NumOPs<<endl;
    
  for(;;){

   // Main learning loop
    MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;

    for(p=0;p<NumPats;p++){ // for each pattern...

      int rand_index = random_gen(NumPats-1);

      // Cal neural network output
     for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
        float in=0;
        for(j=0;j<NumIPs;j++)
          in+=w1[j][i]*x[p][j];      
          h1[i]= (float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn     
      }
      for(i=0;i<NumOPs;i++){ // Cal O/P of output layer
        float in=0;
        for(j=0;j<NumHN1;j++){
          in+=w2[j][i]*h1[j];
        }
        y[i]= (float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
      }
      // Cal error for this pattern
      PatErr=0.0;
      for(i=0;i<NumOPs;i++){
        float err=y[i]-d[p][i]; // actual-desired O/P

        if(err>0)PatErr+=err; else PatErr-=err;
        NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
      }
      if(PatErr<MinErr)MinErr=PatErr;
      if(PatErr>MaxErr)MaxErr=PatErr;
      AveErr+=PatErr;     
      //cout <<setprecision(6)<<setw(6)<<"Error: "<< PatErr;
    }
           
    ItCnt++;
    AveErr/=NumPats;
    float PcntErr = NumErr/float(NumPats) * 100.0;
    cout.setf(ios::fixed|ios::showpoint);
    cout<<setprecision(6)<<setw(6)<<ItCnt<<": "<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<<"pcnterror"<<PcntErr<< setw(12) <<"patterror" << PatErr<<endl;

    if((AveErr<=ObjErr)||(ItCnt==NumIts)) break;
  }// end main testing loop

  // Free memory
  delete h1; delete y; 
  delete ad1; delete ad2;
}

float **Aloc2DAry(int m,int n){
//Allocates memory for 2D array
  float **Ary2D = new float*[m];
  if(Ary2D==NULL){cout<<"No memory!\n";exit(1);}
  for(int i=0;i<m;i++){
	 Ary2D[i] = new float[n];
	 if(Ary2D[i]==NULL){cout<<"No memory!\n";exit(1);}
  }
  return Ary2D;
}

void Free2DAry(float **Ary2D,int n){
//Frees memory in 2D array
  for(int i=0;i<n;i++)
	 delete [] Ary2D[i];
  delete [] Ary2D;
}


//-----------------------------------------------------
//TrainNet for 3 layer NN
//------------------------------------------------------

void TrainNet3(float **x,float **d,int NumIPs,int NumOPs,int NumPats ){
// Trains 3 layer back propagation neural network
// x[][]=>input data, d[][]=>desired output data

  float *h1 = new float[NumHN1]; // O/Ps of hidden layer
  float *h2 = new float[NumHN2]; // O/Ps of hidden layer 2
  
  float *y  = new float[NumOPs]; // O/P of Net
  
  float *ad1= new float[NumHN1]; // HN1 back prop errors
  float *ad2= new float[NumHN2]; // O/P back prop errors
  float *ad3= new float[NumOPs]; // O/P back prop errors

  float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
  int p,i,j;     // for loops indexes
  long ItCnt=0;  // Iteration counter
  long NumErr=0; // Error counter (added for spiral problem)

  ofstream myfile;
  myfile.open ("experiment1.csv");
  myfile << "Train results.\n";
  myfile << "IternationCount,MinErr,AverageErr,MaxError,PercentageError,PatternError,\n";
  

  cout<<"TrainNet3: IP:"<<NumIPs<<" H1:"<<NumHN1<<" OP:"<<NumOPs<<endl;

  // Allocate memory for weights
  w1   = Aloc2DAry(NumIPs,NumHN1);// 1st layer wts
  w11  = Aloc2DAry(NumIPs,NumHN1);
  w111 = Aloc2DAry(NumIPs,NumHN1);
  
  w2   = Aloc2DAry(NumHN1,NumHN2);// 2nd layer wts
  w22  = Aloc2DAry(NumHN1,NumHN2);
  w222 = Aloc2DAry(NumHN1,NumHN2);

  w3   = Aloc2DAry(NumHN2,NumOPs);// 3rd layer wts
  w33  = Aloc2DAry(NumHN2,NumOPs);
  w333 = Aloc2DAry(NumHN2,NumOPs);
    
    
  // Init wts between -0.5 and +0.5
  srand(time(0));
  for(i=0;i<NumIPs;i++)
    for(j=0;j<NumHN1;j++)
    w1[i][j]=w11[i][j]=w111[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN1;i++)
    for(j=0;j<NumHN2;j++)
      w2[i][j]=w22[i][j]=w222[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN2;i++)
    for(j=0;j<NumOPs;j++)
      w3[i][j]=w33[i][j]=w333[i][j]= float(rand())/RAND_MAX - 0.5;

    
  for(;;){// Main learning loop
    MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;

    for(p=0;p<NumPats;p++){ // for each pattern...

      int rand_index = random_gen(NumPats-1);

      // Cal neural network output
      for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
        float in=0;
        for(j=0;j<NumIPs;j++)
          in+=w1[j][i]*x[p][j];
        h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
       // h1[i] = (float)tanh(in);
      }

      for(i=0;i<NumHN2;i++){  //Cal O/P of hidden layer 2
        float in=0;
        for(j=0;j<NumHN1;j++){
          in+=w2[j][i]*h1[j];
        }
        h2[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
       // h2[i] = (float)tanh(in);
       // cout <<"h2" << i << h2[i] <<endl;
      }

      for(i=0;i<NumOPs;i++){ // Cal O/P of output layer

        float in=0;
        for(j=0;j<NumHN2;j++){
          in+=w3[j][i]*h2[j];
      }

      y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
      //  y[i] = (float)tanh(in);

        // cout <<"y" <<"i:" <<i <<"value" <<y[i] << "desiredouput:"<< d[p][i] <<endl;
      }

      // Cal error for this pattern
      PatErr=0.0;
      for(i=0;i<NumOPs;i++){
        float err=y[i]-d[p][i]; // actual-desired O/P
        if(err>0)PatErr+=err; else PatErr-=err;
        NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
        
        //cout << "desiredouput:"<< d[p][i]<<"actual output:" << y[i] <<"NumErr"<< NumErr<<endl;
        //-----------------
       // break;
      }
      if(PatErr<MinErr)MinErr=PatErr;
      if(PatErr>MaxErr)MaxErr=PatErr;
      AveErr+=PatErr;
      
      // Learn pattern with back propagation


      //error correction
       for(i=0;i<NumOPs;i++){ // Modify layer 3(output) wts
      // ad3[i]=(d[p][i]-y[i])*(1-tanh(y[i]))*(1+tanh(y[i]));
        ad3[i]=(d[p][i]-y[i])*y[i]*(1.0-y[i]);
      
      }
      for(i=0;i<NumHN2;i++){ // Modify layer 2 wts
        float err=0.0;
        for(j=0;j<NumOPs;j++)
          err+=ad3[j]*w3[i][j];
        ad2[i] = err*h2[i]*(1.0-h2[i]);
      //  ad2[i] = err * (1-tanh(h2[i]))*(1+tanh(h2[i]));

      }
      for(i=0;i<NumHN1;i++){ // Modify layer 1 wts
        float err=0.0;
        for(j=0;j<NumHN2;j++)
          err+=ad2[j]*w2[i][j];
        
        //ad1[i]=err*(1-tanh(h1[i]))*(1+tanh(h1[i]));
         ad1[i] = err *h1[i]*(1.0-h1[i]);

      }
     
      //------weights correction------------

      for(i=0;i<NumHN1;i++){ // Modify layer 1 wts
        
        for(j=0;j<NumIPs;j++){
          w1[j][i]+=LrnRate * x[p][j] *ad1[i]+
                    Mtm1*(w1[j][i]-w11[j][i])+
                    Mtm2*(w11[j][i]-w111[j][i]);
          w111[j][i]=w11[j][i];
          w11[j][i]=w1[j][i];
        }
      }

      for(i=0;i<NumHN2;i++){ // Modify layer 2 wts
        for(j=0;j<NumHN1;j++){
          w2[j][i]+=LrnRate*h1[j]*ad2[i]+
                    Mtm1*(w2[j][i]-w22[j][i])+
                    Mtm2*(w22[j][i]-w222[j][i]);
          w222[j][i]=w22[j][i];
          w22[j][i]=w2[j][i];
        }
      }

      for(i=0;i<NumOPs;i++){ // Modify layer 3(output) wts
        for(j=0;j<NumHN2;j++){
          w3[j][i]+=LrnRate*h2[j]*ad3[i]+
                    Mtm1*(w3[j][i]-w33[j][i])+
                    Mtm2*(w33[j][i]-w333[j][i]);
          w333[j][i]=w33[j][i];
          w33[j][i]=w3[j][i];
        }
      }
      //end weights correction
   
    }// end for each pattern
      
    ItCnt++;
    AveErr/= NumPats;
    float PcntErr = NumErr/float(NumPats) * 100.0;
    cout.setf(ios::fixed|ios::showpoint);
    cout<<setprecision(6)<<setw(6)<<ItCnt<<":"<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<<PcntErr <<endl;
    myfile << ItCnt <<"," << MinErr << "," << AveErr << "," << MaxErr  <<"," <<  PcntErr << "," << PatErr <<"\n";
 
    if((AveErr<=ObjErr)||(ItCnt==NumIts)) break;


  }// end main learning loop

  myfile.close();

  // Free memory
  delete h1; delete h2;delete y; 
  delete ad1; delete ad2;delete ad3;


}


//-----------------------------------------------------

//TrainNet for 4 layer NN

//------------------------------------------------------

void TrainNet4(float **x,float **d,int NumIPs,int NumOPs,int NumPats ){
// Trains 4 layer back propagation neural network
// x[][]=>input data, d[][]=>desired output data

  float *h1 = new float[NumHN1]; // O/Ps of hidden layer
  float *h2 = new float[NumHN2]; // O/Ps of hidden layer 2
  float *h3 = new float[NumHN3]; // O/Ps of hidden layer 3
  
  float *y  = new float[NumOPs]; // O/P of Net
  
  float *ad1= new float[NumHN1]; // HN1 back prop errors
  float *ad2= new float[NumHN2]; // HN2 back prop errors
  float *ad3= new float[NumHN3]; // HN3 back prop errors
  float *ad4= new float[NumOPs]; // O/P back prop errors

  float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
  int p,i,j;     // for loops indexes
  long ItCnt=0;  // Iteration counter
  long NumErr=0; // Error counter (added for spiral problem)

  cout<<"TrainNet3: IP:"<<NumIPs<<" H1:"<<NumHN1<<" OP:"<<NumOPs<<endl;

  // Allocate memory for weights
  w1   = Aloc2DAry(NumIPs,NumHN1);// 1st layer wts
  w11  = Aloc2DAry(NumIPs,NumHN1);
  w111 = Aloc2DAry(NumIPs,NumHN1);
  
  w2   = Aloc2DAry(NumHN1,NumHN2);// 2nd layer wts
  w22  = Aloc2DAry(NumHN1,NumHN2);
  w222 = Aloc2DAry(NumHN1,NumHN2);

  w3   = Aloc2DAry(NumHN2,NumHN3);// 3rd layer wts
  w33  = Aloc2DAry(NumHN2,NumHN3);
  w333 = Aloc2DAry(NumHN2,NumHN3);

  w4   = Aloc2DAry(NumHN3,NumOPs);// 3rd layer wts
  w44  = Aloc2DAry(NumHN3,NumOPs);
  w444 = Aloc2DAry(NumHN3,NumOPs);
    
    
    
  // Init wts between -0.5 and +0.5
  srand(time(0));
  for(i=0;i<NumIPs;i++)
    for(j=0;j<NumHN1;j++)
    w1[i][j]=w11[i][j]=w111[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN1;i++)
    for(j=0;j<NumHN2;j++)
      w2[i][j]=w22[i][j]=w222[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN2;i++)
    for(j=0;j<NumHN3;j++)
      w3[i][j]=w33[i][j]=w333[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN3;i++)
    for(j=0;j<NumOPs;j++)
      w4[i][j]=w44[i][j]=w444[i][j]= float(rand())/RAND_MAX - 0.5;

 
  for(;;){// Main learning loop
    MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;

    for(p=0;p<NumPats;p++){ // for each pattern...

      int rand_index = random_gen(NumPats-1);

      // Cal neural network output
      for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
        float in=0;
        for(j=0;j<NumIPs;j++)
          in+=w1[j][i]*x[p][j];
        h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
      }

      for(i=0;i<NumHN2;i++){  //Cal O/P of hidden layer 2
        float in=0;
        for(j=0;j<NumHN1;j++){
          in+=w2[j][i]*h1[j];
        }
        h2[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        //cout <<"h2" << i << h2[i] <<endl;

      }

      for(i=0;i<NumHN3;i++){  //Cal O/P of hidden layer 3
        float in=0;
        for(j=0;j<NumHN2;j++){
          in+=w3[j][i]*h2[j];
        }
        h3[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        //cout <<"h2" << i << h2[i] <<endl;

      }
     
      for(i=0;i<NumOPs;i++){ // Cal O/P of output layer
        float in=0;
        for(j=0;j<NumHN3;j++){
          in+=w4[j][i]*h3[j];
        }
        y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
         //cout <<"y" <<"i:" <<i <<"value" <<y[i] <<endl;
      }

      // Cal error for this pattern
      PatErr=0.0;
      for(i=0;i<NumOPs;i++){
        float err=y[i]-d[p][i]; // actual-desired O/P
        if(err>0)PatErr+=err; else PatErr-=err;
        NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
        
        //cout << "desiredouput:"<< d[p][i]<<"error:" << y[i] << "NumErr"<< NumErr<<endl;
        //-----------------
       // break;
      }
      if(PatErr<MinErr)MinErr=PatErr;
      if(PatErr>MaxErr)MaxErr=PatErr;
      AveErr+=PatErr;
  
      // Learn pattern with back propagation

        for(i=0;i<NumOPs;i++){ // Modify layer 4(output) wts
          ad4[i] = (d[p][i]-y[i])*y[i]*(1.0-y[i]);
        
        for(j=0;j<NumHN3;j++){
          w4[j][i]+=LrnRate*h3[j]*ad4[i]+
                    Mtm1*(w4[j][i]-w44[j][i])+
                    Mtm2*(w44[j][i]-w444[j][i]);
          w444[j][i]=w44[j][i];
          w44[j][i]=w4[j][i];
        }
      }
      for(i=0;i<NumHN3;i++){ // Modify layer 3 wts

        float err=0.0;
        for(j=0;j<NumOPs;j++)
          err+=ad4[j]*w4[i][j];

        ad3[i]= err*h3[i]*(1.0-h3[i]);
        
        for(j=0;j<NumHN2;j++){
          w3[j][i]+=LrnRate*h2[j]*ad3[i]+
                    Mtm1*(w3[j][i]-w33[j][i])+
                    Mtm2*(w33[j][i]-w333[j][i]);
          w333[j][i]=w33[j][i];
          w33[j][i]=w3[j][i];
        }
      }

     for(i=0;i<NumHN2;i++){ // Modify layer 2 wts
        float err=0.0;
        for(j=0;j<NumHN3;j++)
          err+=ad3[j]*w3[i][j];
        ad2[i] = err*h2[i]*(1.0-h2[i]);
        
        for(j=0;j<NumHN1;j++){
          w2[j][i]+=LrnRate*h1[j]*ad2[i]+
                    Mtm1*(w2[j][i]-w22[j][i])+
                    Mtm2*(w22[j][i]-w222[j][i]);
          w222[j][i]=w22[j][i];
          w22[j][i]=w2[j][i];
        }
      }
      for(i=0;i<NumHN1;i++){ // Modify layer 1 wts
        float err=0.0;
        for(j=0;j<NumHN2;j++)
          err+=ad2[j]*w2[i][j];
        ad1[i]=err*h1[i]*(1.0-h1[i]);
        
        for(j=0;j<NumIPs;j++){
          w1[j][i]+=LrnRate * x[p][j] *ad1[i]+
                    Mtm1*(w1[j][i]-w11[j][i])+
                    Mtm2*(w11[j][i]-w111[j][i]);
          w111[j][i]=w11[j][i];
          w11[j][i]=w1[j][i];
        }
      } 
    }// end for each pattern
      
    ItCnt++;
    AveErr/= NumPats;
    float PcntErr = NumErr/float(NumPats) * 100.0;
    cout.setf(ios::fixed|ios::showpoint);
    cout<<setprecision(6)<<setw(6)<<ItCnt<<":"<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<<PcntErr<< setw(12) <<endl;
    //cout << setprecision(6)<< "numerrt" << NumErr <<endl;

    if((AveErr<=ObjErr)||(ItCnt==NumIts)) break;
  }// end main learning loop

  // Free memory
   delete h1; delete h2; delete h3; delete y; 
  delete ad1; delete ad2;delete ad3;delete ad4;
}
//-----------------------------------------------------

//TrainNet for 5 layer NN

//------------------------------------------------------

void TrainNet5(float **x,float **d,int NumIPs,int NumOPs,int NumPats ){
// Trains 4 layer back propagation neural network
// x[][]=>input data, d[][]=>desired output data

  float *h1 = new float[NumHN1]; // O/Ps of hidden layer
  float *h2 = new float[NumHN2]; // O/Ps of hidden layer 2
  float *h3 = new float[NumHN3]; // O/Ps of hidden layer 3
  float *h4 = new float[NumHN4]; // O/Ps of hidden layer 4

  float *y  = new float[NumOPs]; // O/P of Net
  
  float *ad1= new float[NumHN1]; // HN1 back prop errors
  float *ad2= new float[NumHN2]; // HN2 back prop errors
  float *ad3= new float[NumHN3]; // HN3 back prop errors
  float *ad4= new float[NumHN4]; // HN4 back prop errors
  float *ad5= new float[NumOPs]; // O/P back prop errors


  float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
  int p,i,j;     // for loops indexes
  long ItCnt=0;  // Iteration counter
  long NumErr=0; // Error counter (added for spiral problem)

  ofstream myfile;
  myfile.open ("experiment1.csv");
  myfile << "Train results.\n";
  myfile << "IternationCount,MinErr,AverageErr,MaxError,PercentageError,PatternError,\n";
  

  cout<<"TrainNet5: IP:"<<NumIPs<<" H1:"<<NumHN1 <<"H2:" <<NumHN2 << "H3:" <<NumHN3 <<"H4:"<< NumHN4 <<" OP:"<<NumOPs<<endl;

  // Allocate memory for weights
  w1   = Aloc2DAry(NumIPs,NumHN1);// 1st layer wts
  w11  = Aloc2DAry(NumIPs,NumHN1);
  w111 = Aloc2DAry(NumIPs,NumHN1);
  
  w2   = Aloc2DAry(NumHN1,NumHN2);// 2nd layer wts
  w22  = Aloc2DAry(NumHN1,NumHN2);
  w222 = Aloc2DAry(NumHN1,NumHN2);

  w3   = Aloc2DAry(NumHN2,NumHN3);// 3rd layer wts
  w33  = Aloc2DAry(NumHN2,NumHN3);
  w333 = Aloc2DAry(NumHN2,NumHN3);

  w4   = Aloc2DAry(NumHN3,NumHN4);// 4rth layer wts
  w44  = Aloc2DAry(NumHN3,NumHN4);
  w444 = Aloc2DAry(NumHN3,NumHN4);

  w5   = Aloc2DAry(NumHN4,NumOPs);// 5th layer wts
  w55  = Aloc2DAry(NumHN4,NumOPs);
  w555 = Aloc2DAry(NumHN4,NumOPs);
    
  // Init wts between -0.5 and +0.5
  srand(time(0));
  for(i=0;i<NumIPs;i++)
    for(j=0;j<NumHN1;j++)
    w1[i][j]=w11[i][j]=w111[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN1;i++)
    for(j=0;j<NumHN2;j++)
      w2[i][j]=w22[i][j]=w222[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN2;i++)
    for(j=0;j<NumHN3;j++)
      w3[i][j]=w33[i][j]=w333[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN3;i++)
    for(j=0;j<NumHN4;j++)
      w4[i][j]=w44[i][j]=w444[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN4;i++)
    for(j=0;j<NumOPs;j++)
      w5[i][j]=w55[i][j]=w555[i][j]= float(rand())/RAND_MAX - 0.5;

    
  for(;;){// Main learning loop
    MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;

    for(p=0;p<NumPats;p++){ // for each pattern...

      int rand_index = random_gen(NumPats-1);

      // Cal neural network output
      for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
        float in=0;
        for(j=0;j<NumIPs;j++)
          in+=w1[j][i]*x[p][j];
        h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        //h1[i] = (float)tanh(in);

      }

      for(i=0;i<NumHN2;i++){  //Cal O/P of hidden layer 2
        float in=0;
        for(j=0;j<NumHN1;j++){
          in+=w2[j][i]*h1[j];
        }
        //h2[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        h2[i] = (float)tanh(in);
        //cout <<"h2" << i << h2[i] <<endl;

      }

      for(i=0;i<NumHN3;i++){  //Cal O/P of hidden layer 3
        float in=0;
        for(j=0;j<NumHN2;j++){
          in+=w3[j][i]*h2[j];
        }
        //h3[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        h3[i] = (float)tanh(in);

        //cout <<"h2" << i << h2[i] <<endl;

      }

       for(i=0;i<NumHN4;i++){  //Cal O/P of hidden layer 4
        float in=0;
        for(j=0;j<NumHN3;j++){
          in+=w4[j][i]*h3[j];
        }
        //h4[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        h4[i] = (float)tanh(in);

        //cout <<"h2" << i << h2[i] <<endl;
      }
     
      for(i=0;i<NumOPs;i++){ // Cal O/P of output layer
        float in=0;
        for(j=0;j<NumHN4;j++){
          in+=w5[j][i]*h4[j];
        }
        //y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
          y[i] = (float)tanh(in);
         //cout <<"y" <<"i:" <<i <<"value" <<y[i] <<endl;
      }

      // Cal error for this pattern
      PatErr=0.0;
      for(i=0;i<NumOPs;i++){
        float err=y[i]-d[p][i]; // actual-desired O/P
        if(err>0)PatErr+=err; else PatErr-=err;
        NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
        
        //cout << "desiredouput:"<< d[p][i]<<"error:" << ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5)) << "NumErr"<< NumErr<<endl;
        //-----------------
       // break;
      }
      if(PatErr<MinErr)MinErr=PatErr;
      if(PatErr>MaxErr)MaxErr=PatErr;
      AveErr+=PatErr;
    
      
      // Learn pattern with back propagation

      //error correction 
      for(i=0;i<NumOPs;i++){ // Modify layer 4(output) wts
          ad5[i] = (d[p][i]-y[i])*(1-tanh(y[i]))*(1+tanh(y[i]));
      }

      for(i=0;i<NumHN4;i++){ // Modify layer 4(output) wts
        float err=0.0;
        for(j=0;j<NumOPs;j++)
          err+=ad5[j]*w5[i][j];
        ad4[i] = err*(1-tanh(h4[i]))*(1+tanh(h4[i]));
      }
      for(i=0;i<NumHN3;i++){ // Modify layer 3 wts

        float err=0.0;
        for(j=0;j<NumHN4;j++)
          err+=ad4[j]*w4[i][j];

        ad3[i]= err*(1-tanh(h3[i]))*(1+tanh(h3[i]));
      }

     for(i=0;i<NumHN2;i++){ // Modify layer 2 wts
        float err=0.0;
        for(j=0;j<NumHN3;j++)
          err+=ad3[j]*w3[i][j];
        ad2[i] = err*(1-tanh(h2[i]))*(1+tanh(h2[i]));
      }
      for(i=0;i<NumHN1;i++){ // Modify layer 1 wts
        float err=0.0;
        for(j=0;j<NumHN2;j++)
          err+=ad2[j]*w2[i][j];
        ad1[i]=err*h1[i]*(1.0-h1[i]);
       }
       //-------update wights--------------------------

      for(i=0;i<NumOPs;i++){ // Modify layer 5(output) wts
        
        for(j=0;j<NumHN4;j++){
          w5[j][i]+=LrnRate*h4[j]*ad5[i]+
                    Mtm1*(w5[j][i]-w55[j][i])+
                    Mtm2*(w55[j][i]-w555[j][i]);
          w555[j][i]=w55[j][i];
          w55[j][i]=w5[j][i];
        }
      }

      for(i=0;i<NumHN4;i++){ // Modify layer 4(output) wts
        
        for(j=0;j<NumHN3;j++){
          w4[j][i]+=LrnRate*h3[j]*ad4[i]+
                    Mtm1*(w4[j][i]-w44[j][i])+
                    Mtm2*(w44[j][i]-w444[j][i]);
          w444[j][i]=w44[j][i];
          w44[j][i]=w4[j][i];
        }
      }
      for(i=0;i<NumHN3;i++){ // Modify layer 3 wts
        for(j=0;j<NumHN2;j++){
          w3[j][i]+=LrnRate*h2[j]*ad3[i]+
                    Mtm1*(w3[j][i]-w33[j][i])+
                    Mtm2*(w33[j][i]-w333[j][i]);
          w333[j][i]=w33[j][i];
          w33[j][i]=w3[j][i];
        }
      }

     for(i=0;i<NumHN2;i++){ // Modify layer 2 wts
        
        for(j=0;j<NumHN1;j++){
          w2[j][i]+=LrnRate*h1[j]*ad2[i]+
                    Mtm1*(w2[j][i]-w22[j][i])+
                    Mtm2*(w22[j][i]-w222[j][i]);
          w222[j][i]=w22[j][i];
          w22[j][i]=w2[j][i];
        }
      }
      for(i=0;i<NumHN1;i++){ // Modify layer 1 wts
        
        for(j=0;j<NumIPs;j++){
          w1[j][i]+=LrnRate * x[p][j] *ad1[i]+
                    Mtm1*(w1[j][i]-w11[j][i])+
                    Mtm2*(w11[j][i]-w111[j][i]);
          w111[j][i]=w11[j][i];
          w11[j][i]=w1[j][i];
        }
      }
      //weights correction

    }// end for each pattern
      
    ItCnt++;
    AveErr/= NumPats;
    float PcntErr = NumErr/float(NumPats) * 100.0;
    cout.setf(ios::fixed|ios::showpoint);
    cout<<setprecision(6)<<setw(6)<<ItCnt<<":"<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<< setw(12)  << PcntErr<<endl;
    //cout << setprecision(6)<< "numerrt" << NumErr <<endl;
    myfile << ItCnt <<"," << MinErr << "," << AveErr << "," << MaxErr  <<"," <<  PcntErr << "," << PatErr <<"\n";
     

    if((AveErr<=ObjErr)||(ItCnt==NumIts)) break;
  }// end main learning loop
    myfile.close();
  // Free memory
  delete h1; delete h2; delete h3; delete h4; delete y; 
  delete ad1; delete ad2;delete ad3;delete ad4;delete ad5;
}



//------------------------------------------------------------


//------------------------------------------------------------

void TestNet3 (float **x,float **d,int NumIPs,int NumOPs,int NumPats ){
  
  //x[][]=>input data, d[][]=>desired output data

  float *h1 = new float[NumHN1]; // O/Ps of hidden layer
  float *h2 = new float[NumHN2]; // O/Ps of hidden layer

  float *y  = new float[NumOPs]; // O/P of Net

  float *ad1= new float[NumHN1]; // HN1 back prop errors
  float *ad2= new float[NumHN2]; // HN1 back prop errors
  float *ad3= new float[NumOPs]; // O/P back prop errors

  float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
  int p,i,j;     // for loops indexes
  long ItCnt=0;  // Iteration counter
  long NumErr=0; // Error counter (added for spiral problem)

  cout<<"TestNet3: IP:"<<NumIPs<<" H1:"<<NumHN1<<" OP:"<<NumOPs<<endl;
    
  for(;;){

   // Main learning loop
    MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;

    for(p=0;p<NumPats;p++){ // for each pattern...

      int rand_index = random_gen(NumPats-1);

      // Cal neural network output
     for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
        float in=0;
        for(j=0;j<NumIPs;j++)
          in+=w1[j][i]*x[p][j];      
          h1[i]= (float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn     
      }
      for(i=0;i<NumHN2;i++){ // Cal O/P of hidden layer 1
        float in=0;
        for(j=0;j<NumHN1;j++)
          in+=w2[j][i]*h1[j];
          h2[i]= (float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn     
      }
      for(i=0;i<NumOPs;i++){ // Cal O/P of output layer
        float in=0;
        for(j=0;j<NumHN2;j++){
          in+=w3[j][i]*h2[j];
        }
        y[i]= (float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
      }
      // Cal error for this pattern
      PatErr=0.0;
      for(i=0;i<NumOPs;i++){
        float err=y[i]-d[p][i]; // actual-desired O/P

        if(err>0)PatErr+=err; else PatErr-=err;
        NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
      }
      if(PatErr<MinErr)MinErr=PatErr;
      if(PatErr>MaxErr)MaxErr=PatErr;
      AveErr+=PatErr;     
      //cout <<setprecision(6)<<setw(6)<<"Error: "<< PatErr;
    }
           
    ItCnt++;
    AveErr/=NumPats;
    float PcntErr = NumErr/float(NumPats) * 100.0;
    cout.setf(ios::fixed|ios::showpoint);
    cout<<setprecision(6)<<setw(6)<<ItCnt<<": "<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<<PcntErr<< setw(12) <<"patterror" << PatErr<<endl;

    if((AveErr<=ObjErr)||(ItCnt==NumIts)) break;
  }// end main testing loop

  // Free memory
  delete h1;delete h2;  delete y; 
  delete ad1; delete ad3;
}

