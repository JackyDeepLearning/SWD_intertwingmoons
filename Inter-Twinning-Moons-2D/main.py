import os
import Demo
import torch
import argparse
import MoonData
import numpy as np
import matplotlib.pyplot as plt
import Different_Module

def main():

    para       = para_args.parashow
    mode       = para['mode']
    seed       = para['seed']
    Type       = para['Type']
    channel    = para['channel']
    parameter  = para['parameter']
    interation = para['interation']
    learn_rate = para['learning-rate']

    xs  , ys  , xt          = MoonData.dataload(mode,parameter)
    x_gp, y_gp, xtx, ytx, z = MoonData.grid_print(mode,xs)

    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

    g, f1, f2, step_f1, step_f2, step_gen, step_all = Demo.AllToyNet(channel,learn_rate)

    for step in range(interation):

        g, f1, f2, step_f1, step_f2, step_gen, step_all, loss = Different_Module.DifModule(Type, g, f1, f2, step_f1, step_f2, step_gen, step_all, xs, ys, xt)

        if step % 1000 == 0:

            print("Iteration: %d / %d" % (step, 10000))
            print('Step A loss: %d', loss[0])
            print('Step B loss: %d', loss[1])
            print('Step C loss: %d', loss[2])

            with torch.no_grad():
                feature_z    = g(z)
                classifier_z = (f2(feature_z).cpu().numpy() > 0.5).astype(np.float32)

            classifier_z = classifier_z.reshape(x_gp.shape)
            figure_plt   = plt.figure()
            plt.contourf(x_gp, y_gp, classifier_z, cmap=plt.cm.copper_r, alpha=0.9)
            plt.scatter(xs[:,0], xs[:,1], c=ys.reshape((len(xs))), cmap=plt.cm.coolwarm, alpha=0.8)
            plt.scatter(xt[:,0], xt[:,1], color='green', alpha=0.7)
            plt.text(xtx, ytx, 'Iter:' + str(step), fontsize=14, color='#FFD700', bbox=dict(facecolor='dimgray', alpha=0.7))
            plt.axis('off')
            figure_plt.savefig(Type + '_pytorch_iter' + str(step) + '.png', bbox_inches='tight', pad_inches=0, dpi=100, transparent=True)
            plt.close()


if __name__ == '__main__':
    parashow  = {'mode': 0, 'channel': 20, 'seed': 1234,
                 'parameter': [150, -3, 0.4, 5.3, 9, 0.6],
                 'interation':10001, 'learning-rate': 0.02,
                 'Type': 'Source_Only'}
    Argument  = argparse.ArgumentParser()
    Argument.add_argument('--parashow', type=dict, help='parashow', default=parashow)
    para_args = Argument.parse_args()
    main()
