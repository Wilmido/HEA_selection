## Mining High Entropy Alloys as Electrocatalyst for Oxygen Reduction Reaction by Machine Learning
This is a demo repository for high entropy alloys(HEAs) experiments for "<b>【学科交叉项目】神经网络机器学习辅助筛选高熵合金催化剂(\[Interdisciplinary Project\] Neural network machine learning assisted screening of high entropy alloy catalysts)</b>" in "Innovation and Entrepreneurship Project for College Students"(2021), HX2021037

I build up a regression model to predict the OH* adsorption energy of HEAs, and a WGAN-GP(Wasserstein GAN using gradient penalty) to generate HEA compositions


## Dependecies
The prominent packages are:
* numpy
* pandas
* matplotlib
* scikit-learn
* pytorch 1.8.1

To install all the dependencies quickly and easily, you should use __pip__ install `requirements.txt`
```python
pip install -r requirements.txt
```


## Get Started
To train the model, you can simply use the following command, and you will get a checkpoint:
```
# training a model for downstream tasks
python K_fold.py
```

Obtaining the plot of __MAE__ and __RMSE__ compared with DFT-calculated adsorption energy

```
# training a model for downstream tasks, you need to update the checkpoint path first! 
python main.py
```

*Pretrained Models*
 ---
You can also just simply use the checkpoint I have provided in <a href="HEA_selection/checkpoint/9_700epochs_5_model.pth">checkpoint/9_700epochs_5_model.pth</a>.


## T-SNE
Visualize the data, and the features processed by the model. 
```
python t_SNE.py
```

## Generate HEAs
You can switch the mode to choose whether to train the regression model. The result of loss plot demonstrates that the training process of GAN is not good :(
```
python Joint_training.py
```

![Figure_1](https://user-images.githubusercontent.com/71449089/161955711-a5e78e40-a2df-4e1f-8768-3045bf9f8024.png)


## Reference
https://github.com/jol-jol/neural-network-design-of-HEA

https://github.com/Zeleni9/pytorch-wgan

