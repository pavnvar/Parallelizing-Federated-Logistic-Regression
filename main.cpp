#include<iostream>
#include<vector>
#include<math.h>
#include<fstream>
#include<string>

#define feature_num 5
using namespace std;

vector<double> expectedOutput;
vector<vector<double>> inputValues;

vector<double> Validation_expectedOutput;
vector<vector<double>> Validation_inputValues;


double e = 2.71828;
long epoch = 100;
double weight[feature_num]={0};
double learningRate = 0.001;


double activation(double z);
void updateWeight(double predicted, double expected, vector<double> inputs);
void calculateAccuracy();
void test();
void datainput(vector<double> &expectedOutput1, vector<vector<double>> &inputValues1);
void datastandarization(vector<vector<double>> &inputValues2);
int main()
{
  cout<<"Training Data File Name: ";
  datainput(expectedOutput,inputValues);
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
      cout<<"Epoch "<<(100-epoch)<<" ";
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
  calculateAccuracy();

  test();
  return 0 ;
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
  cout<<"Accuracy is: "<<(totalCorrect * 100) / totalCases<<"%"<<endl;
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
  cout<<"Validation Data File Name: ";
  datainput(Validation_expectedOutput, Validation_inputValues);
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

void datainput(vector<double> &expectedOutput1, vector<vector<double>> &inputValues1)
{
  // open a file in write mode.
  ifstream myfile;
  string line,filename;
  cin>>filename;
  myfile.open(filename);  
  if (!myfile)
  {                 
    cout<<"Error while creating the file\n";    
  }
  else
  {
    cout<<"File created successfully\n";          
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