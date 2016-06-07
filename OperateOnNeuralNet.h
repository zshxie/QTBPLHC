#ifndef OPERATEONNEURALNET_H
#define OPERATEONNEURALNET_H




//#ifndef __OPERATEONNEURALNET_H__

//#define __OPERATEONNEURALNET_H__



#include "NeuralNet.h"

#define NEURALNET_VERSION 1.0

#define RESAMPLE_LEN 4



class COperateOnNeuralNet{

private:

         /*network*/

         CNeuralNet *m_oNetWork;


         /*network'sparameter*/

         int m_nInput;

         int m_nOutput;

         int m_nNeuronsPerLyr;

         int m_nHiddenLayer;


         /*trainingconfiguration*/

         int m_nMaxEpoch;  // max training epoch times

         double m_dMinError; // error threshold

         double m_dLearningRate;



         /*dinamiccurrent parameter*/

         int m_nEpochs;

         double m_dErr;  //mean error of one epoch(m_dErrorSum/(num-of-samples * num-of-output))

         bool m_bStop; //control whether stop or not during the training

         vector<double> m_vecError;   //record each epoch'straining error, used for drawing error curve



public:

         COperateOnNeuralNet();

         ~COperateOnNeuralNet();



         void SetNetWorkParameter(int nInput, int nOutput, int nNeuronsPerLyr, intnHiddenLayer);

         bool CreatNetWork();

         void SetTrainConfiguration(int nMaxEpoch, double dMinError, double dLearningRate);

         void SetStopFlag(bool bStop) { m_bStop = bStop; }



         double GetError(){ return m_dErr; }

         int GetEpoch(){ return m_nEpochs; }

         int GetNumNeuronsPerLyr(){ return m_nNeuronsPerLyr; }



         bool Train(vector<iovector>& SetIn, vector<iovector>& SetOut);

         bool SaveTrainResultToFile(const char* lpszFileName, boolbCreate);

         bool LoadTrainResultFromFile(const char* lpszFileName, DWORDdwStartPos);

         int Recognize(CString strPathName, CRect rt, double&dConfidence);

};

/*

* Can be used when saving or readingtraining result.

*/

struct NEURALNET_HEADER{

         DWORD dwVersion;  //version imformation



         /*initial parameters*/

         int m_nInput;  //number of inputs

         int m_nOutput; //number of outputs

         int m_nNeuronsPerLyr; //unit number of hidden layer

         int m_nHiddenLayer; //hidden layer, not including the output layer



         /*training configuration*/

         int m_nMaxEpoch;  // max training epoch times

         double m_dMinError; // error threshold

         double m_dLearningRate;



         /*dinamiccurrent parameter*/

         int m_nEpochs;

         double m_dErr;  //mean error of oneepoch(m_dErrorSum/(num-of-samples * num-of-output))

};

#endif //__OPERATEONNEURALNET_H__







#endif // OPERATEONNEURALNET_H
