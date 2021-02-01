# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.


## Summary
The dataset, related to [this csv file](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv), contains information retrieved by marketing calls about the customers of a bank that opened a bank account. 

To do it we used Azure ML first with HyperDrive(0.9138088012139606) and then with Auto ML (0.916206373292868). The results of these comparisons shown that we got the best performance with AutoML.

![pipeline structure](https://video.udacity-data.com/topher/2020/September/5f639574_creating-and-optimizing-an-ml-pipeline/creating-and-optimizing-an-ml-pipeline.png)

## Scikit-learn Pipeline
The pipeline includes in order:
- a *train.py* script to apply regression to a specific dataset;
- define a Parameter Sampler;
- choose the Bandit Policy; 
- tune a HyperDrive with some config such as the metric (Accuracy) and the parameters above.
the result is a Hyperparameter model.

**Parameter Sampler**
I used a RandomParameterSampling because it's support early stop and because it fit well with HyperDrive. 

```
RandomParameterSampling(
    parameter_space={
                        "C": uniform(0.0, 1.0), 
                        "max_iter": choice(50,100,150,200,250)
                    })
```

**Bandit Policy**
I choosed the Bandit Policy because it terminate runs when the metric choosed is not near the specified slack/factor.
```
 BanditPolicy(evaluation_interval=2, slack_factor=0.000000001, delay_evaluation=10)
```

## AutoML
The AutoML pipeline tested 22 different Pipeline and the best one was VotingEnsemble:
```
VotingEnsemble                                0:01:18       0.9162    0.9162
```

## Pipeline comparison
The comparison between HyperDrive and AutoML show how the use of only 2 classifier of the first get a worst performance compared with the 24 of the second.

## Future work
For a future improve of this work can be done the follow actions:
- perform more execution time for AutoML to try with more classifier;
- try to use the other Parameter Sampling such as Bayesian. 

## Proof of cluster clean up
At the end of the notebook execution I delete the compute cluster used.
![delete cluster](https://www.dropbox.com/s/txo35ysjawkw88o/Screenshot%202021-02-01%20at%2023.05.25.png?dl=1)
