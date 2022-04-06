import os
import torch
import argparse
from model.networks import MyModel
from utils.data_loader import BasicDataset
from utils.Trainlogger import Logger
from utils.evaluation import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_ratio',type=float, default=0.5, help='Split dataset')
    parser.add_argument('--flag', type=bool, default=False, help='whether to use atoms augment')
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    parser.add_argument('--epochs', type=int, default=1000, help='Numbers of Epoch to train')
    parser.add_argument('--k',type=int, default=5, help='k-fold')
    args = parser.parse_args()

    mylogger = Logger(args,filename='Evaluation_log')
    mylogger.logger.info('Device:'+torch.cuda.get_device_name(0))


    #================= Training ================#

    split_RATIO = args.split_ratio
    if args.flag:
        DATA_PATH = 'data/best_result.csv'
        dataset = BasicDataset(DATA_PATH)
    else:
        dataset = BasicDataset()
    num_feature = dataset.get_feature_number()
    # x_train, y_train, x_test, y_test = dataset.get_data(device, split_RATIO)
    x ,y = dataset.get_data(device, 1)

    mylogger.logger.info(f'The number of featrue: {num_feature}')

    model = MyModel(num_feature=num_feature).to(device)
    file_name = f'{num_feature}_{args.epochs}epochs_{args.k}_model.pth'    
    # file_name = '9_700epochs_5_model.pth'
    file_path = args.save_path + file_name
    if os.path.isfile(file_path):
        Checkpoint = torch.load(file_path)
        model.load_state_dict(Checkpoint['model_state_dict'])
        
        mylogger.logger.info('Loading model successfully!')
    else:
        mylogger.logger.error(f'No {file_name} file!\nPlease run K_fold.py first!')							
        os._exit(0)


    with torch.no_grad():
        model.eval()
        """ train_pred = model(x_train).detach().cpu().numpy()
        y_train = y_train.detach().cpu().numpy()
        train_MAE, train_RMSE = evaluate(train_pred, y_train) """

        """ test_pred = model(x_test).detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
        test_MAE, test_RMSE = evaluate(test_pred, y_test) """
        pred = model(x).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        MAE, RMSE = evaluate(pred, y)
   
    #$ MAE and RMSE
    mylogger.logger.info(f"Train MAE: {MAE:.4f}	RMSE: {RMSE:.4f}")

    # mylogger.logger.info(f"Train MAE: {train_MAE:.4f}	RMSE: {train_RMSE:.4f}")
    # mylogger.logger.info(f"Test MAE: {test_MAE:.4f}	RMSE: {test_RMSE:.4f}")

    # torch.save(model.state_dict(), 'sample.pth')


    #======== Ploting results =========#
    import matplotlib.pyplot as plt
    # initiate figure
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})

    # show training set and testing set
    # ax.scatter(y_train, train_pred, 15, color='blue', marker='.', label='training set')
    # ax.scatter(y_test, test_pred, 15, color='red',  marker='x', label='testing set')
    ax.scatter(y, pred, 10, color='red',  marker='x', label='data point')
    # show MAE and RMSE
    """ ax.text(-0.8, -2.0, 'training (%i points)\nMAE=%.2f eV RMSE=%.2f eV'%
                (len(x_train), train_MAE, train_RMSE),fontsize=10)
    ax.text(-0.8, -2.0-0.3, 'testing (%i points)\nMAE=%.2f eV RMSE=%.2f eV'%
                    (len(x_test), test_MAE, test_RMSE),fontsize=10) """
    ax.text(-0.8, -2.0, '(%i points)\nMAE=%.3f eV \nRMSE=%.3f eV'%(len(x), MAE, RMSE),fontsize=10)
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

    # plt.show()

    #? save image
    plt.savefig(f'result/plot.pdf', format='pdf',bbox_inches='tight', dpi=700)
