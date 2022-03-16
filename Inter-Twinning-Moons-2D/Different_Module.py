import LossType
import numpy as np

def DifModule(mode, genarator, classifier1, classifier2, step_c1, step_c2, step_gen, step_all, xs, ys, xt):

    loss = np.zeros((3,1))

    if mode == 'SWD':

        step_gen.zero_grad()
        step_c1.zero_grad()
        step_c2.zero_grad()

        feature  = genarator(xs)
        predict1 = classifier1(feature)
        predict2 = classifier2(feature)
        Step_A   = LossType.loss_type(5, predict1, predict2, ys)
        loss[0]  = Step_A.item()

        Step_A.backward()
        step_all.step()
        step_all.zero_grad()

        deta_feature  = genarator(xs)
        deta_feature  = deta_feature.detach()
        deta_predict1 = classifier1(deta_feature)
        deta_predict2 = classifier2(deta_feature)
        Step_All      = LossType.loss_type(5, deta_predict1, deta_predict2, ys)

        target_feature  = genarator(xt)
        target_feature  = target_feature.detach()
        target_predict1 = classifier1(target_feature)
        target_predict2 = classifier2(target_feature)
        Step_B          = Step_All - LossType.loss_type(4, target_predict1, target_predict2, 0)
        loss[1]         = Step_B.item()

        Step_B.backward()
        step_c1.step()
        step_c2.step()
        step_gen.zero_grad()

        target_feature  = genarator(xt)
        target_predict1 = classifier1(target_feature)
        target_predict2 = classifier2(target_feature)
        Step_C          = LossType.loss_type(4, target_predict1, target_predict2, 0)
        loss[2]         = Step_C.item()

        Step_C.backward()
        step_gen.step()

    elif mode == 'MCD':

        step_gen.zero_grad()
        step_c1.zero_grad()
        step_c2.zero_grad()

        feature  = genarator(xs)
        predict1 = classifier1(feature)
        predict2 = classifier2(feature)
        Step_A   = LossType.loss_type(5, predict1, predict2, ys)
        loss[0]  = Step_A.item()

        Step_A.backward()
        step_all.step()
        step_all.zero_grad()

        deta_feature  = genarator(xs)
        deta_feature  = deta_feature.detach()
        deta_predict1 = classifier1(deta_feature)
        deta_predict2 = classifier2(deta_feature)
        Step_All = LossType.loss_type(5, deta_predict1, deta_predict2, ys)

        target_feature  = genarator(xt)
        target_feature  = target_feature.detach()
        target_predict1 = classifier1(target_feature)
        target_predict2 = classifier2(target_feature)
        Step_B          = Step_All - LossType.loss_type(3, target_predict1, target_predict2, 0)
        loss[1]         = Step_B.item()

        Step_B.backward()
        step_c1.step()
        step_c2.step()
        step_gen.zero_grad()

        target_feature  = genarator(xt)
        target_predict1 = classifier1(target_feature)
        target_predict2 = classifier2(target_feature)
        Step_C          = LossType.loss_type(3, target_predict1, target_predict2, 0)
        loss[2]         = Step_C.item()

        Step_C.backward()
        step_gen.step()

    elif mode == 'Source_Only':

        feature = genarator(xs)
        predict1 = classifier1(feature)
        predict2 = classifier2(feature)
        Step_A = LossType.loss_type(5, predict1, predict2, ys)
        loss[0] = Step_A.item()

        Step_A.backward()
        step_all.step()
        step_all.zero_grad()

    return genarator, classifier1, classifier2, step_c1, step_c2, step_gen, step_all, loss



