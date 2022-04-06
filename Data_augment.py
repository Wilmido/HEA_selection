from model import MyModel, Generator, Discriminator
from utils import get_data,get_dataloader, gradient_penalty, get_disc_loss, get_gen_loss
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

"""
0.0,1.0,0.0,8.0,1.0,
0.0,0.0,0.0,6.0,1.0,
1.0,1.0,1.5,11.0,0.0,
0.0,2.0,3.0,9.0,0.0,
0.0,2.0,3.0,12.0,0.0,
1.0,1.0,1.5,12.0,0.0,
0.0,0.0,0.0,9.0,0.0,
1.0,2.0,4.5,7.0,0.0,
0.0,1.0,0.0,12.0,0.0,
1.0,1.0,1.5,10.0,0.0,
1.0,2.0,4.5,7.0,0.0,
0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,9.0,0.0,
0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
z_dim = 64
lr = 0.001
c_lambda = 1
disc_repeats = 1 # number of times to update the discriminator per generator update
gen_repeats = 1
reg_repeats = 8

def get_noise(n_sample, z_dim, device):
	"""
    Function for creating nosie vectors: Given the dimensions (n_sample, z_dim)
	n_sample : the number of samples to generate, a scalar
    z_dim: the dimension of the nosie vector,a scalar
	"""
	return torch.randn(n_sample, z_dim).to(device) 


dataloader, _ = get_dataloader(1)

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.SGD(gen.parameters(), lr=lr / 2)
disc = Discriminator().to(device)
disc_opt = torch.optim.SGD(disc.parameters(), lr=lr)


TRAINING_RATIO = 0.5
train_dl, test_dl = get_dataloader(TRAINING_RATIO)

R_model = MyModel().to(device)
optimizer = torch.optim.Adam(R_model.parameters(),lr=1e-5, weight_decay=0.0004)
criterion = nn.MSELoss()

cur_step = 0
generator_losses = []
discriminator_losses = []


for epoch in range(epochs):
    for real, _ in dataloader:
        real = real.to(device)
        cur_batch_size = len(real)

        #==== train discriminator =====#
        mean_iteration_disc_loss = 0
        for _ in range(disc_repeats):
            disc_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device)
            fake = gen(fake_noise)
            fake_pred = disc(fake)
            real_pred = disc(real)

            epsilon = torch.rand(cur_batch_size, 1, 1, requires_grad=True).to(device)
            gp = gradient_penalty(disc, real, fake, epsilon)
            disc_loss = get_disc_loss(fake_pred, real_pred, gp, c_lambda)

            mean_iteration_disc_loss += disc_loss.item() / disc_repeats
            # upadate gradients
            disc_loss.backward(retain_graph=True)
            # update optimizer
            disc_opt.step()
        discriminator_losses += [mean_iteration_disc_loss]

        #==== update generator =====#
        mean_iteration_gen_loss = 0
        for _ in range(gen_repeats):

            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device)
            fake_2 = gen(fake_noise_2)
            fake_pred2 = disc(fake_2)

            gen_loss = get_gen_loss(fake_pred2)
            mean_iteration_gen_loss += gen_loss.item() / gen_repeats
            
            gen_loss.backward()
            gen_opt.step()

        generator_losses += [gen_loss.item()]

        #======== train regression modle =========#
        
    R_model.train()
    for _ in range(reg_repeats):
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            y_pred = R_model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 


        if cur_step % 100 == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-100:]) / 100
            disc_mean = sum(discriminator_losses[-100:]) / 100
            print(f"Step {cur_step}  Generator loss: {gen_mean:.4f} \
                Discriminator loss: {disc_mean:.4f}")

        cur_step += 1



    x_train, y_train, x_test, y_test = get_data(TRAINING_RATIO)
    with torch.no_grad():
        R_model.eval()
        train_pred = R_model(x_train.to(device)).cpu().detach().numpy()
        y_train = y_train.detach().numpy()
        test_pred = R_model(x_test.to(device)).cpu().detach().numpy()
        y_test = y_test.detach().numpy()

        #$ MAE and RMSE
        train_MAE = np.sum(np.abs(y_train - train_pred)) / len(x_train)
        test_MAE = np.sum(np.abs(y_test - test_pred)) / len(x_test)
        train_RMSE = np.sqrt(np.sum((y_train - train_pred)**2) / len(x_train))
        test_RMSE = np.sqrt(np.sum((y_test - test_pred)**2) / len(x_test))

    if epoch % 20 == 0:
        print(f"train MAE: {train_MAE:.4f}	RMSE: {train_RMSE:.4f}")
        print(f"test MAE:  {test_MAE:.4f}	RMSE: {test_RMSE:.4f}")



    #======= Visualization code ======#
    if  epoch == epochs - 1:
        
        """ step_bins = 20
        num_examples = (len(generator_losses) // step_bins) * step_bins
        plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Geneartor Loss"
        )

        plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label= "Discriminator Loss"
            ) """
        
        plt.plot(range(cur_step), generator_losses, label="Generator Loss")
        plt.plot(range(cur_step), discriminator_losses, label="Discriminator Loss")        
        plt.legend()
        plt.show()



#===== ploting =====#
import matplotlib.pyplot as plt
# initiate figure
fig, ax = plt.subplots()
plt.rcParams.update({'font.size': 12})
# show training set and testing set
ax.scatter(y_train, train_pred, 15, color='blue', marker='.', label='training set')
ax.scatter(y_test, test_pred, 15, color='red',  marker='x', label='testing set')

# show MAE and RMSE
ax.text(-0.8, -2.0, 'training (%i points)\nMAE=%.2f eV RMSE=%.2f eV'%
			(len(x_train), train_MAE, train_RMSE),fontsize=10)
ax.text(-0.8, -2.0-0.3, 'testing (%i points)\nMAE=%.2f eV RMSE=%.2f eV'%
				(len(x_test), test_MAE, test_RMSE),fontsize=10)

# plot solid diagonal line
ax.plot([-2.5,0.5], [-2.5,0.5], 'k', label=r'$\Delta E_{\mathrm{pred}} = \Delta E_{\mathrm{DFT}}$')
# plot dashed diagonal lines 0.15 eV above and below solid diagonal line
ax.plot([-2.5,0.5], [-2.35,0.65], 'k--', label=r'$\pm %.2f \mathrm{eV}$'%(0.15))
ax.plot([-2.5,0.5], [-2.65,0.35], 'k--')


# set legend sytle

ax.legend(fontsize=10, loc='upper left')
#.get_frame().set_edgecolor('k') 

# set style of labels
plt.xlabel(r'DFT-calculated $\Delta E_{\mathrm{OH}}-\Delta E_{\mathrm{OH, Pt(111)}}$ (eV)')
plt.ylabel('Neural network-predicted\n'+
           r'$\Delta E_{\mathrm{OH}}-\Delta E_{\mathrm{OH, Pt(111)}}$ (eV)')
plt.xlim([-2.5, 0.5]); plt.ylim([-2.5,0.5])
plt.box(on=True)
plt.tick_params(direction='in', right=True, top=True)

plt.show()


"""
#! 理想情况下，正常的Generator的loss曲线应该是不断往0靠近的下降的抛物线
#! 而 Discriminator 的loss曲线应该是在0附近震荡。 

#? 现在的情况就是discriminator的loss曲线出现震荡，认为可能是discriminator太强导致的。
使用低的C_lambda

优先训练Discriminator
有研究认为Batch Normalization对于Generator有负面作用
一般都认为Batch Normalization对于Discriminator有积极作用

在GAN中尽量避免使用池化层MaxPool，AvgPool。使用Leaky-ReLU代替ReLU

1.更大的kernel，更多的filter

#= 不要early stopping
当Discriminator的loss接近0时，Generator就很难学到任何东西了，就需要终止训练。

#! 这里考虑minmaxscaler而不是standardsclaer是因为这里的数据点分布0比较多。而且Generator最后一层的
#! 输出激活函数是tanh,限制最后生成的样本是在0，1。

使用了全新的模型结构后，discriminator的结构存在优化的空间

"""
        