#include<iostream>
#include<vector>
#include<math.h>
#include<fstream>
#include<string>
#include <mpi.h>

#define feature_num 30
#define training_nodes 4

using namespace std;

vector<double> expectedOutput;
vector<vector<double>> inputValues;

vector<double> Validation_expectedOutput;
vector<vector<double>> Validation_inputValues;
vector<string> dataFiles = {"data.csv"};
string validationFile = "validation.csv";

double e;
long epoch;
double weight[feature_num];
double nodeReceivedTrainedWeights[training_nodes][feature_num] = {0};
double learningRate;
int globalrank;

char processor_name[MPI_MAX_PROCESSOR_NAME];
int name_len;

double activation(double z);
void updateWeight(double predicted, double expected, vector<double> inputs);
void calculateAccuracy();
void test();
void datainput(vector<double> &expectedOutput1, vector<vector<double>> &inputValues1, string filename);
void datastandarization(vector<vector<double>> &inputValues2);
void trainlogisticregressionmodel();
void initializeParameters();
void FederatedLearningWithMPIReducePrimitive();
void FederatedLearningWithSendReceivePrimitive();

int main()
{
  initializeParameters();
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &globalrank);
  MPI_Get_processor_name(processor_name, &name_len);
  cout << "Process ID is " << globalrank << " on node " << processor_name << endl;
  //FederatedLearningWithMPIReducePrimitive();
  FederatedLearningWithSendReceivePrimitive();
  MPI_Finalize();
  return 0 ;
}

void FederatedLearningWithMPIReducePrimitive()
{
  if(globalrank != 0){
    trainlogisticregressionmodel();
    calculateAccuracy();
  }
  MPI_Reduce( (globalrank == 0) ? MPI_IN_PLACE : weight, weight, feature_num, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(globalrank == 0){
    for (int i = 0; i < feature_num; ++i)
    {
        weight[i] = weight[i]/4;
        cout << weight[i] << " ";
    }  
    cout << endl;
    test();
  }
}


void FederatedLearningWithSendReceivePrimitive()
{

  if(globalrank == 0)
  {
      //Receive weights from all the prcoess ranked 1,2,3,4
      for(int sender_node=0 ; sender_node<4; sender_node++)
      {
        MPI_Recv(nodeReceivedTrainedWeights[sender_node], feature_num, MPI_DOUBLE, sender_node+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Received weights from process  "<< sender_node+1;
        for (int i = 0; i < feature_num; ++i)
        {
            cout << nodeReceivedTrainedWeights[sender_node][i] << " ";
            weight[i] += nodeReceivedTrainedWeights[sender_node][i];
        }  
      }
      
      for (int i = 0; i < feature_num; ++i)
      {
          weight[i] = weight[i]/4;
          cout << weight[i] << " ";
      }  
      cout << endl;
      test();
  }

  if(globalrank != 0){
      trainlogisticregressionmodel();
      //Send all the weights to process 0
      MPI_Send(weight, feature_num, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      calculateAccuracy();
  }
}

void trainlogisticregressionmodel()
{
  string dataFile = "data.csv";
  //cout << "Data file name is "<< dataFile;
  datainput(expectedOutput, inputValues, dataFile);
  datastandarization(inputValues);
  for(int i = 0;i < feature_num;i++)
  {
    weight[i] = 0.01;
  }
  //check values for proper input into vectors
  // for(int i = 0; i < inputValues.size(); i++) 
  // {
  //   for(int j = 0; j < feature_num; j++) 
  //   {
  //     cout<<inputValues[i][j]<<" ";
  //   }
  //   cout<<endl;
  // }
  // cout<<"inputValues.size(); = "<<inputValues.size()<<endl; 
  // inputValues.size() = 469
  // cout<<"inputValues[0].size(); = "<<inputValues[0].size()<<endl; 
  // inputValues[0].size() = 31
  
  while(epoch--) 
  {
      //cout<<"Epoch "<<(100-epoch)<<" ";
      calculateAccuracy();
      for(int i = 0; i < inputValues.size(); i++) 
      {
          double predictedValue, z = 0;
          for(int j = 1; j < inputValues[0].size(); j++) 
          {
            z += weight[j-1] * inputValues[i][j];
          }
          predictedValue = activation(z);
          updateWeight(predictedValue, expectedOutput[i], inputValues[i]);
      }
  }
}

void initializeParameters()
{
  Validation_expectedOutput.clear();
  Validation_inputValues.clear();
  expectedOutput.clear();
  inputValues.clear();
  e = 2.71828;
  epoch = 100;
  for(int feature_index=0; feature_index<feature_num; feature_index++)
  {
    weight[feature_index] = 0;
  }
  learningRate = 0.001;
}

double activation(double z) 
{
  return 1/(1 + pow(e, (-1 * z)));
}

void calculateAccuracy() 
{

  long totalCorrect = 0, totalCases = inputValues.size();

  for(int i = 0; i < totalCases; i++) 
  {
    double predictedValue, z = 0;
    for(int j = 1; j < inputValues[0].size(); j++) 
    {
      z += inputValues[i][j] * weight[j-1];
    }

    predictedValue = activation(z);
    predictedValue = (predictedValue<0.5)? 0:1; // 0 = malignant, 1 = benign
    if(predictedValue == expectedOutput[i]) 
    {
      totalCorrect++;
    }
  }
  //cout<<"Accuracy is: "<<(totalCorrect * 100) / totalCases<<"%"<<endl;
}

void updateWeight(double predictedValue, double expectedOutput, vector<double> inputValue) 
{
  for(int i = 0; i < inputValue.size(); i++) 
  {
    double gradientDescent;
    gradientDescent = (predictedValue - expectedOutput) * inputValue[i];
    weight[i] = weight[i] - (learningRate * gradientDescent);
  }
}

void test() 
{
  double z = 0;
  int totalCorrect = 0;
  //cout<<"Validation Data File Name: " << validationFile;
  datainput(Validation_expectedOutput, Validation_inputValues, validationFile);
  datastandarization(Validation_inputValues);
  for(int i = 0; i < Validation_inputValues.size(); i++) 
  {
    double predictedValue, z = 0;
    for(int j = 1; j < Validation_inputValues[0].size(); j++) 
    {
      z += Validation_inputValues[i][j] * weight[j-1];
    }

    predictedValue = activation(z);
    predictedValue = (predictedValue<0.5)? 0:1;
    if(predictedValue == Validation_expectedOutput[i]) 
    {
      totalCorrect++;
    }
  }
  cout<<"Validation Accuracy is: "<<(totalCorrect * 100) / Validation_inputValues.size()<<"%"<<endl;
}

void datainput(vector<double> &expectedOutput1, vector<vector<double>> &inputValues1, string filename)
{
  // open a file in write mode.
  ifstream myfile;
  string line;
  myfile.open(filename);  
  if (!myfile)
  {                 
    cout<<"Error while opening the file\n";    
  }
  else
  {
    //cout<<"File opened  successfully\n";          
  }
  getline(myfile,line);//to skip the first catagory sentense
  while(getline(myfile,line)) // input data into C++
  {
    //cout<<line<<endl;
    int i = 0;
    int token; //the values to be read from file
    vector<double> inputRow;
    double output;

    inputRow.clear();

    for(token = 0; token < feature_num+1; token++) 
    {
      double value;
      string val = "";

      while(line[i] != ',') 
      {
        val += line[i];        
        i++; //increment the character
        if(i== line.size()) //out the loop in the last parameter
        {
          break;
        }
      }
      i++; //move beyond the comma
      value = stod(val);
      inputRow.push_back(value);
    }
    i++; //move beyond the comma

    string outputStr = "";
        
    outputStr = line[0];
    output = stod(outputStr);

    expectedOutput1.push_back(output);
    inputValues1.push_back(inputRow);
  }
  myfile.close();
}

void datastandarization(vector<vector<double>> &inputValues2)
{
  for(int i = 1; i < feature_num+1; i++) //calculate mean value
  {
    double sum = 0;
    double mean = 0;
    double var_sum = 0;
    double SD = 0;
    // cout<<"inputValues2.size(); = "<<inputValues2.size()<<endl; 
    // cout<<"inputValues2[0].size(); = "<<inputValues2[0].size()<<endl; 
    for(int j = 0; j < inputValues2.size(); j++) 
    {
      sum += inputValues2[j][i];
    }
    mean = sum/inputValues2.size();

    for(int j = 0; j < inputValues2.size(); j++) // calculate variance and SD
    {
      var_sum += pow(inputValues2[j][i] - mean, 2);
    }
    SD = sqrt(var_sum/inputValues2.size());

    for(int j = 0; j < inputValues2.size(); j++) // standarization
    {
      inputValues2[j][i] = ((inputValues2[j][i]-mean)/SD);
    }
  }
  // for(int i = 0; i < inputValues2.size(); i++) 
  // {
  //   for(int j = 0; j < feature_num+1; j++) 
  //   {
  //     cout<<inputValues2[i][j]<<" ";
  //   }
  //   cout<<endl;
  // }
}