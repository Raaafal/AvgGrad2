import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


average_gradient_of_linear_layers_enhancement=False#for method=1 or method=2
average_gradient_of_nonlinear_layers_enhancement=True#for method=1 or method=2
average_gradient_of_loss=False#for method=1 and iter_count = 2, seems to not affect





denominator_mul = 0.9  # deprecated
step_is_fraction_of_optimizer_denominator = 0.  # deprecated
class NN(nn.Module):
    nonlinear_complex=()#deprecated#(torch.nn.Softmax,torch.nn.LogSoftmax)#(torch.nn.Softmax,torch.nn.LogSoftmax)#todo: delete all classes that doesn't exist among model.layers
    nonlinear_simple=(torch.nn.GELU,torch.nn.ELU,torch.nn.Sigmoid,torch.nn.Tanh)#(torch.nn.Sigmoid,torch.nn.Tanh)#(torch.nn.Sigmoid,torch.nn.ReLU)

    step_factor=1#0.347#0.336#0.39#0.9#0.39#0.85
    gradient_factor=1#0.622#0.621#0.53#0.05#0.53#0.05

    gradient_factor_simple_layers=1#0.354#0.25#0.25#0.5#0.5#0.25#1.

    small_num = (torch.finfo(torch.float32).eps)  # **0.75#sys.float_info.epsilon**0.5
    stats=None
    input_cached=None

    gradient_factor_how_many_times_faster_decreases_than_increases=4
    gradient_factor_increase_multipler=(2**(1/100))#gradient may change maximally 2 times through 300 updates (300 batches)
    gradient_factor_decrease_multipler=1/gradient_factor_increase_multipler**gradient_factor_how_many_times_faster_decreases_than_increases
    def loss_signal(self,higher:bool):
        return
        #self.step_factor=1.
        if higher:
            #self.gradient_factor=self.gradient_factor*self.gradient_factor_decrease_multipler
            self.step_factor = self.step_factor * self.gradient_factor_decrease_multipler
        else:
            #self.gradient_factor=self.gradient_factor*self.gradient_factor_increase_multipler
            #self.gradient_factor = min(self.gradient_factor * self.gradient_factor_increase_multipler, 1.)
            self.step_factor = min(self.step_factor * self.gradient_factor_increase_multipler, 1.)

        self.step_factor=min(self.step_factor,0.25)#0.25
        #self.gradient_factor=self.step_factor
        self.gradient_factor_simple_layers=self.gradient_factor

    def __init__(self,layers):
        super(NN, self).__init__()

        self.layers=layers
        self.layer_list=nn.ModuleList(self.layers)

        self.intermediate_outputs=[None for i in range(len(self.layers))]
        self.intermediate_outputs_grad=[None for i in range(len(self.layers))]


        if step_is_fraction_of_optimizer_denominator!=0.:
            for l in self.layers:
                if hasattr(l,'weight'):
                    l.weight.denominator=0.*l.weight
                    l.bias.denominator = 0. * l.bias
                    # l.weight.denominator.requires_grad=False
                    # l.bias.denominator.requires_grad = False
        #torch.autograd.set_detect_anomaly(True)


    @torch.no_grad()#copy of gradient_calc
    def gradient_calc_last_layer2(self,layer_input,layer_input_candidate,layer_input_candidate_grad,layer_out,layer_out_candidate_gradient,layer,loss_fn,target):
        input_mod=layer_input.detach().clone()
        # layer_input_candidate=layer_input_candidate.detach().clone()
        # layer_input_candidate+=0.001
        reshaped_input = layer_input.reshape((layer_input.size()[0], -1)).detach()
        reshaped_input_mod=input_mod.reshape((input_mod.size()[0],-1)).detach()
        reshaped_input_candidate=layer_input_candidate.reshape((input_mod.size()[0],-1)).detach()
        #reshaped_input_grad=layer_input.grad.reshape((layer_input.grad.size()[0],-1)).detach()
        reshaped_input_grad=layer_input_candidate_grad.reshape((layer_input_candidate_grad.size()[0],-1)).detach()

        reshaped_output_candidate_gradient=layer_out_candidate_gradient.reshape((layer_out_candidate_gradient.size()[0],-1)).detach()
        #last_index=-1
        #out=torch.empty_like(layer_input)
        for i in range(reshaped_input_mod.size()[1]):
            if i>0:
                reshaped_input_mod[:,i-1]=reshaped_input[:,i-1].detach()


            out1 = layer(input_mod)
            loss1 = loss_fn(out1, target)

            #reshaped_input_mod[:, i] = reshaped_input[:, i].detach() + 0.0001 #test
            #reshaped_input_mod[:, i] = reshaped_input_candidate[:, i].detach() #original one
            reshaped_input_mod[:, i] = reshaped_input_candidate[:, i].detach()*self.step_factor+reshaped_input[:, i]*(1.-self.step_factor)

            out = layer(input_mod)
            # out=nn.LogSoftmax(dim=1)(input_mod.detach().clone())
            loss2 = loss_fn(out, target)

            for j in range(reshaped_input_grad.size()[0]):
                diff=reshaped_input_mod[j,i]-reshaped_input[j,i]
                if torch.abs(diff)>self.small_num:
                    #reshaped_input_grad[j,i]=torch.sum((out[j].detach()-layer_out[j].detach())*layer_out.grad[j].detach()/diff)
                    reshaped_input_grad[j, i] =reshaped_input_grad[j, i]*(1-self.gradient_factor)+self.gradient_factor* (loss2 - loss1) / diff.detach()
                #else:
                #    print("Diff<=very small number")
    @torch.no_grad()
    def gradient_calc_last_layer(self, layer_input, layer_input_candidate, layer_input_candidate_grad, layer_out,
                      layer_out_candidate_gradient, layer,loss_fn,target):#doesn't work well
        input_mod = layer_input.detach().clone()
        # layer_input_candidate=layer_input_candidate.detach().clone()
        # layer_input_candidate+=0.001
        reshaped_input = layer_input.reshape((layer_input.size()[0], -1)).detach()
        reshaped_input_mod = input_mod.reshape((input_mod.size()[0], -1)).detach()
        reshaped_input_candidate = layer_input_candidate.reshape((input_mod.size()[0], -1)).detach()
        # reshaped_input_grad=layer_input.grad.reshape((layer_input.grad.size()[0],-1)).detach()
        reshaped_input_grad = layer_input_candidate_grad.reshape((layer_input_candidate_grad.size()[0], -1)).detach()

        reshaped_output_candidate_gradient = layer_out_candidate_gradient.reshape(
            (layer_out_candidate_gradient.size()[0], -1)).detach()
        # last_index=-1
        # out=torch.empty_like(layer_input)


        for i in range(reshaped_input_mod.size()[1]):
            if i > 0:
                reshaped_input_mod[:, i - 1] = reshaped_input[:, i - 1].detach()

            out1 = layer(input_mod)
            loss1 = loss_fn(out1, target)

            # reshaped_input_mod[:, i] = reshaped_input[:, i].detach() + 0.0001 #test
            #reshaped_input_mod[:, i] = reshaped_input_candidate[:, i].detach() #original one
            reshaped_input_mod[:, i] = reshaped_input_candidate[:, i].detach() * 0.5 + reshaped_input[:, i] * 0.5


            out = layer(input_mod)
            # out=nn.LogSoftmax(dim=1)(input_mod.detach().clone())
            loss2 = loss_fn(out,target)

            for j in range(reshaped_input_grad.size()[0]):
                diff = reshaped_input_mod[j, i] - reshaped_input[j, i]
                if torch.abs(diff) > self.small_num:
                    # reshaped_input_grad[j,i]=torch.sum((out[j].detach()-layer_out[j].detach())*layer_out.grad[j].detach()/diff)
                    #reshaped_input_grad[j, i] = reshaped_input_grad[j, i] * 0.5 + 0.5 * (loss2-loss1) / diff.detach()
                    reshaped_input_grad[j, i] =reshaped_input_grad[j, i]*0.5+0.5* (loss2 - loss1) / diff.detach()
                # else:
                #     print("Diff<=very small number")


    @torch.no_grad()
    def gradient_calc_simple(self,layer_input,layer_input_candidate,layer_input_candidate_grad,layer_out,layer_out_candidate_gradient,layer):
        # #input_mod=layer_input.detach().clone()
        # # layer_input_candidate=layer_input_candidate.detach().clone()
        # # layer_input_candidate+=0.001
        # reshaped_input = layer_input.reshape((layer_input.size()[0], -1)).detach()
        # #reshaped_input_mod=input_mod.reshape((input_mod.size()[0],-1)).detach()
        # reshaped_input_candidate=layer_input_candidate.reshape((layer_input_candidate.size()[0],-1)).detach()
        # #reshaped_input_grad=layer_input.grad.reshape((layer_input.grad.size()[0],-1)).detach()
        # reshaped_input_grad=layer_input_candidate_grad.reshape((layer_input_candidate_grad.size()[0],-1)).detach()
        #
        # reshaped_output_candidate_gradient=layer_out_candidate_gradient.reshape((layer_out_candidate_gradient.size()[0],-1)).detach()
        # #last_index=-1
        # #out=torch.empty_like(layer_input)
        out = layer(layer_input_candidate)

        layer_input_candidate_grad[:]=torch.where(torch.abs(layer_input_candidate-layer_input)>self.small_num,
                                                  layer_input_candidate_grad*(1.-self.gradient_factor_simple_layers)+self.gradient_factor_simple_layers*(out-layer_out)*layer_out_candidate_gradient/(layer_input_candidate-layer_input),
                                                  layer_input_candidate_grad)

        # for i in range(reshaped_input.size()[1]):#that code maybe works
        #
        #     for j in range(reshaped_input_grad.size()[0]):
        #         diff=reshaped_input_candidate[j,i]-reshaped_input[j,i]
        #         if torch.abs(diff)>self.small_num:
        #             #reshaped_input_grad[j,i]=torch.sum((out[j].detach()-layer_out[j].detach())*layer_out.grad[j].detach()/diff)
        #             reshaped_input_grad[j,i]=reshaped_input_grad[j,i].detach()*(1.-self.gradient_factor)+self.gradient_factor*((out[j,i].detach()-layer_out[j,i].detach())*reshaped_output_candidate_gradient[j,i].detach()/diff.detach())
        #         #else:
        #         #    print("Diff<=very small number")
        #         #else default gradient value is kept
    @torch.no_grad()
    def gradient_calc(self,layer_input,layer_input_candidate,layer_input_candidate_grad,layer_out,layer_out_candidate_gradient,layer):
        input_mod=layer_input.detach().clone()
        # layer_input_candidate=layer_input_candidate.detach().clone()
        # layer_input_candidate+=0.001
        reshaped_input = layer_input.reshape((layer_input.size()[0], -1)).detach()
        reshaped_input_mod=input_mod.reshape((input_mod.size()[0],-1)).detach()
        reshaped_input_candidate=layer_input_candidate.reshape((input_mod.size()[0],-1)).detach()
        #reshaped_input_grad=layer_input.grad.reshape((layer_input.grad.size()[0],-1)).detach()
        reshaped_input_grad=layer_input_candidate_grad.reshape((layer_input_candidate_grad.size()[0],-1)).detach()

        reshaped_output_candidate_gradient=layer_out_candidate_gradient.reshape((layer_out_candidate_gradient.size()[0],-1)).detach()
        #last_index=-1
        #out=torch.empty_like(layer_input)
        for i in range(reshaped_input_mod.size()[1]):
            if i>0:
                reshaped_input_mod[:,i-1]=reshaped_input[:,i-1].detach()

            #reshaped_input_mod[:, i] = reshaped_input[:, i].detach() + 0.0001 #test
            #reshaped_input_mod[:, i] = reshaped_input_candidate[:, i].detach() #original one
            reshaped_input_mod[:, i] = reshaped_input_candidate[:, i].detach()*self.step_factor+reshaped_input[:, i]*(1.-self.step_factor)

            out=layer(input_mod)
            #out=nn.LogSoftmax(dim=1)(input_mod.detach().clone())

            for j in range(reshaped_input_grad.size()[0]):
                diff=reshaped_input_mod[j,i]-reshaped_input[j,i]
                if torch.abs(diff)>self.small_num:
                    #reshaped_input_grad[j,i]=torch.sum((out[j].detach()-layer_out[j].detach())*layer_out.grad[j].detach()/diff)
                    reshaped_input_grad[j,i]=reshaped_input_grad[j,i]*(1.-self.gradient_factor)+self.gradient_factor*torch.sum((out[j].detach()-layer_out[j].detach())*reshaped_output_candidate_gradient[j].detach()/diff.detach())
                # else:
                #     print("Diff<=very small number")
                #else default gradient value is kept

            #reshaped_input_grad[:, i]=1
            #last_index=i

        #reshaped_input_mod[:, reshaped_input_mod.size()[1]-1] = reshaped_input[:, reshaped_input_mod.size()[1]-1].detach()
        #reshaped_input_grad._version-=1


    def backward_grad_correction_with_weight_change2(self,loss,model2,weight_change=True,accumulate_gradients=False,gradient_only_modification=False):
        #start_ind = len(self.intermediate_outputs) - 2 if loss_fn==F.cross_entropy else len(self.intermediate_outputs) - 1
        start_ind = len(self.intermediate_outputs) - 1
        ind = start_ind#len(self.intermediate_outputs) - 1
        grad = torch.autograd.grad(outputs=loss,
                                   inputs=self.intermediate_outputs[start_ind],
                                   # inputs=self.intermediate_outputs[0],
                                   grad_outputs=None
                                   , retain_graph=True)
        self.intermediate_outputs_grad[ind] = grad[0]

        if average_gradient_of_loss:
            self.intermediate_outputs_grad[ind]=0.5*(grad[0]+model2.intermediate_outputs_grad[ind])
        #print(model2.intermediate_outputs_grad[ind])
        #enhanced=False#True
        # if enhanced:
            # # aa=loss_fn(self.intermediate_outputs[ind],target,reduction='none')
            # # aa=zip(self.intermediate_outputs[ind],target).apply_(lambda a:a*a)
            # loss_tensor = torch.zeros(self.intermediate_outputs[ind].size())
            # loss_tensor2 = torch.zeros(model2.intermediate_outputs[ind].size())
            # for i in range(self.intermediate_outputs[ind].size()[0]):
            #     # self.intermediate_outputs[ind][i]=loss_fn(self.intermediate_outputs[ind][i],target[i],reduction='none')
            #     loss_tensor[i] = loss_fn(self.intermediate_outputs[ind][i], target[i], reduction='none')
            #     #loss_tensor[i]=-self.intermediate_outputs[ind][i]*torch.log(target[i])
            #     loss_tensor2[i] = loss_fn(model2.intermediate_outputs[ind][i], target[i], reduction='none')
            # print(str(torch.sum(loss_tensor)- loss_fn(self.intermediate_outputs[ind][i], target[i], reduction='sum')))
            # with torch.no_grad():
            #     # self.intermediate_outputs_grad[ind][:] = torch.where(
            #     #     torch.abs(self.intermediate_outputs[ind] - model2.intermediate_outputs[ind]) > self.small_num,
            #     #     self.intermediate_outputs_grad[ind][:] * (
            #     #                 1. - self.gradient_factor_simple_layers) + self.gradient_factor_simple_layers *
            #     #                 (loss_tensor - loss_tensor2) / (self.intermediate_outputs[ind] - model2.intermediate_outputs[ind]),
            #     #     self.intermediate_outputs_grad[ind][:])
            #     self.intermediate_outputs_grad[ind][:] = torch.where(
            #         torch.abs(self.intermediate_outputs[ind] - model2.intermediate_outputs[ind]) > self.small_num,
            #         (loss_tensor - loss_tensor2) / (self.intermediate_outputs[ind] - model2.intermediate_outputs[ind]),
            #         self.intermediate_outputs_grad[ind][:])
            # grad = torch.autograd.grad(outputs=model2_loss,
            #                            inputs=model2.intermediate_outputs[start_ind],
            #                            # inputs=self.intermediate_outputs[0],
            #                            grad_outputs=None
            #                            , retain_graph=True)
            # #model2.intermediate_outputs_grad[ind] = grad[0]
            # self.intermediate_outputs_grad[ind][:]=.5*(self.intermediate_outputs_grad[ind][:]+grad[0][:])

        for ind in range(start_ind, -1, -1):
            if hasattr(self.layers[ind], 'weight'):
                # if average_gradient_of_linear_layers_enhancement:
                #     self.layers[ind].weight.requires_grad = False
                #     self.layers[ind].bias.requires_grad = False
                #     self.layers[ind].weight[:] = 0.5*(self.layers[ind].weight.detach() + model2.layers[ind].weight.detach())
                #     self.layers[ind].bias[:] = 0.5*(self.layers[ind].bias.detach() + model2.layers[ind].bias.detach())
                #     # self.layers[ind].weight[:] = 0.5 * (
                #     #             -self.layers[ind].weight.detach() + 3. * model2.layers[ind].weight.detach())
                #     # self.layers[ind].bias[:] = 0.5 * (
                #     #             -self.layers[ind].bias.detach() + 3. * model2.layers[ind].bias.detach())
                #     self.layers[ind].weight.requires_grad = True
                #     self.layers[ind].bias.requires_grad = True

                if ind>0:
                    self.intermediate_outputs[ind] = self.layers[ind](self.intermediate_outputs[ind-1])
                else:
                    self.intermediate_outputs[ind]=self.layers[ind](self.input_cached)

                grad_outputs = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
                                                   inputs=[self.layers[ind].weight,self.layers[ind].bias],
                                                   # inputs=self.intermediate_outputs[0],
                                                   grad_outputs=self.intermediate_outputs_grad[ind]
                                                   , retain_graph=True)
                if not average_gradient_of_linear_layers_enhancement:
                    if accumulate_gradients and self.layers[ind].weight.grad is not None:
                        self.layers[ind].weight.grad += grad_outputs[0]
                        self.layers[ind].bias.grad+=grad_outputs[1]
                    else:
                        self.layers[ind].weight.grad = grad_outputs[0]
                        self.layers[ind].bias.grad = grad_outputs[1]
                else:
                    grad_outputs2 = torch.autograd.grad(outputs=model2.intermediate_outputs[ind],
                                                       inputs=(model2.layers[ind].weight, model2.layers[ind].bias),
                                                       # inputs=self.intermediate_outputs[0],
                                                       grad_outputs=self.intermediate_outputs_grad[ind]
                                                       , retain_graph=True)
                    if accumulate_gradients and self.layers[ind].weight.grad is not None:
                        self.layers[ind].weight.grad += 0.5*(grad_outputs[0]+grad_outputs2[0])
                        self.layers[ind].bias.grad+=0.5*(grad_outputs[1]+grad_outputs2[1])
                    else:
                        self.layers[ind].weight.grad = 0.5 * (grad_outputs[0] + grad_outputs2[0])
                        self.layers[ind].bias.grad = 0.5 * (grad_outputs[1] + grad_outputs2[1])


                #self.layers[ind].weight = model2.layers[ind].weight#.clone().detach()
                #self.layers[ind].bias = model2.layers[ind].bias#.clone().detach()

                #self.layers[ind].weight.requires_grad=False
                #self.layers[ind].bias.requires_grad=False
                #self.layers[ind].weight[:]=1#torch.abs(self.layers[ind].weight-model2.layers[ind].weight)*torch.sign(self.layers[ind].weight.grad)
                #self.layers[ind].bias[:]=1#torch.abs(self.layers[ind].bias-model2.layers[ind].bias)*torch.sign(self.layers[ind].bias.grad)
                # self.layers[ind].weight.requires_grad = True
                # self.layers[ind].bias.requires_grad = True

                #model2.layers[ind].weight.grad=grad_outputs[0].detach()#.clone()
                #model2.layers[ind].bias.grad=grad_outputs[1].detach()

                # if ind > 0:
                #     grad_layers = torch.autograd.grad(outputs=self.layers[ind].weight,
                #                                       inputs=self.intermediate_outputs[ind - 1],
                #                                       # inputs=self.intermediate_outputs[0],
                #                                       grad_outputs=self.layers[ind].weight.grad
                #                                       , retain_graph=True)
                #     self.intermediate_outputs[ind - 1].grad = grad_layers[0]

                # if ind>0 and average_gradient_of_linear_layers_enhancement:
                #     self.layers[ind].weight.requires_grad = False
                #     self.layers[ind].bias.requires_grad = False
                #     self.layers[ind].weight[:] = 0.5*(self.layers[ind].weight.detach() + model2.layers[ind].weight.detach())
                #     self.layers[ind].bias[:] = 0.5*(self.layers[ind].bias.detach() + model2.layers[ind].bias.detach())
                #     # self.layers[ind].weight[:] = 0.5 * (
                #     #             -self.layers[ind].weight.detach() + 3. * model2.layers[ind].weight.detach())
                #     # self.layers[ind].bias[:] = 0.5 * (
                #     #             -self.layers[ind].bias.detach() + 3. * model2.layers[ind].bias.detach())
                #     self.layers[ind].weight.requires_grad = True
                #     self.layers[ind].bias.requires_grad = True
                #
                #     self.intermediate_outputs[ind] = self.layers[ind](self.intermediate_outputs[ind - 1])
            if ind > 0:
                #self.intermediate_outputs[ind]+=0.
                #self.intermediate_outputs[ind - 1]+=0.


                grad_layers = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
                                                  inputs=self.intermediate_outputs[ind - 1],
                                                  # inputs=self.intermediate_outputs[0],
                                                  grad_outputs=self.intermediate_outputs_grad[ind]
                                                  , retain_graph=True,
                                                  # allow_unused=True
                                                  )
                self.intermediate_outputs_grad[ind - 1] = grad_layers[0]

                if average_gradient_of_linear_layers_enhancement and not (average_gradient_of_nonlinear_layers_enhancement and isinstance(self.layers[ind], self.nonlinear_simple)):
                    grad_layers2 = torch.autograd.grad(outputs=model2.intermediate_outputs[ind],
                                                      inputs=model2.intermediate_outputs[ind - 1],
                                                      # inputs=self.intermediate_outputs[0],
                                                      grad_outputs=self.intermediate_outputs_grad[ind]
                                                      , retain_graph=True)
                    self.intermediate_outputs_grad[ind - 1] = 0.5*(self.intermediate_outputs_grad[ind - 1]+grad_layers2[0])

                # if ind==len(self.intermediate_outputs)-1:
                #     self.gradient_calc(self.intermediate_outputs[ind - 1],self.intermediate_outputs[ind - 1],
                #                        self.intermediate_outputs[ind],self.layers[ind])
                # pass

                #enhanced=True
                if average_gradient_of_nonlinear_layers_enhancement:

                    if isinstance(self.layers[ind], self.nonlinear_complex):#deprecated
                        # self.gradient_calc(model2.intermediate_outputs[ind - 1], self.intermediate_outputs[ind - 1],
                        #                    model2.intermediate_outputs[ind],self.intermediate_outputs[ind].grad, self.layers[ind])

                        # self.gradient_calc(model2.intermediate_outputs[ind - 1], self.intermediate_outputs[ind - 1],
                        #                    self.intermediate_outputs_grad[ind - 1],
                        #                    model2.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                        #                    self.layers[ind])
                        log=False
                        if log:
                            print("----")
                            print(self.intermediate_outputs_grad[ind - 1])
                        self.gradient_calc(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                                           self.intermediate_outputs_grad[ind - 1],
                                           self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                                           self.layers[ind])
                        # self.gradient_calc_last_layer2(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                        #                    self.intermediate_outputs_grad[ind - 1],
                        #                    self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                        #                    self.layers[ind],loss_fn,target)
                        # self.intermediate_outputs_grad[ind - 1]=0.5*self.intermediate_outputs_grad[ind - 1]+\
                        #                                         model2.intermediate_outputs[ind - 1].grad * 0.5
                        # self.gradient_calc_last_layer(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                        #                    self.intermediate_outputs_grad[ind - 1],
                        #                    self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                        #                    self.layers[ind],loss_fn,target)
                        if log:
                            print(self.intermediate_outputs_grad[ind - 1])
                        #self.intermediate_outputs_grad[ind - 1]=self.intermediate_outputs_grad[ind - 1]*torch.norm(model2.intermediate_outputs_grad[ind - 1],1)/torch.norm(self.intermediate_outputs_grad[ind - 1],1)

                    elif isinstance(self.layers[ind], self.nonlinear_simple):
                        self.gradient_calc_simple(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                                           self.intermediate_outputs_grad[ind - 1],
                                           self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                                           self.layers[ind])
                        #self.intermediate_outputs_grad[ind - 1]=self.intermediate_outputs_grad[ind - 1]*torch.norm(model2.intermediate_outputs_grad[ind - 1],1)/torch.norm(self.intermediate_outputs_grad[ind - 1],1)

                #self.intermediate_outputs[ind - 1].grad.detach()

            if (weight_change or gradient_only_modification) and hasattr(self.layers[ind], 'weight'):
                self.layers[ind].weight.requires_grad=False
                self.layers[ind].bias.requires_grad=False
                # if average_gradient_of_linear_layers_enhancement:
                #     #(2.*...) is because...
                #     self.layers[ind].weight[:]-=2.*torch.abs(self.layers[ind].weight.detach()-model2.layers[ind].weight.detach())*torch.sign(self.layers[ind].weight.grad.detach())
                #     self.layers[ind].bias[:]-=2.*torch.abs(self.layers[ind].bias.detach()-model2.layers[ind].bias.detach())*torch.sign(self.layers[ind].bias.grad.detach())
                # else:
                if gradient_only_modification:
                    self.layers[ind].weight.grad[:] = torch.abs(
                        model2.layers[ind].weight.grad.detach()) * torch.sign(
                        self.layers[ind].weight.grad.detach())
                    self.layers[ind].bias.grad[:] = torch.abs(
                        model2.layers[ind].bias.grad.detach()) * torch.sign(
                        self.layers[ind].bias.grad.detach())
                else:
                    if step_is_fraction_of_optimizer_denominator!=0.:
                        #self.layers[ind].weight.denominator=self.layers[ind].weight.grad.detach()*self.layers[ind].weight.grad.detach()+denominator_mul*self.layers[ind].weight.denominator.detach()
                        #self.layers[ind].bias.denominator = self.layers[ind].bias.grad.detach() * self.layers[ind].bias.grad.detach()+denominator_mul*self.layers[ind].bias.denominator.detach()
                        # self.layers[ind].weight[:] -= torch.min(step_is_fraction_of_optimizer_denominator*torch.sqrt(self.layers[ind].weight.denominator.detach()),torch.abs(
                        #     self.layers[ind].weight.detach() - model2.layers[ind].weight.detach())) * torch.sign(
                        #     self.layers[ind].weight.grad.detach())
                        # self.layers[ind].bias[:] -= torch.min(step_is_fraction_of_optimizer_denominator*torch.sqrt(self.layers[ind].bias.denominator.detach()),torch.abs(
                        #     self.layers[ind].bias.detach() - model2.layers[ind].bias.detach())) * torch.sign(
                        #     self.layers[ind].bias.grad.detach())
                        diff_w=self.layers[ind].weight.grad.detach()-model2.layers[ind].weight.grad.detach()
                        self.layers[ind].weight.denominator=diff_w*diff_w+denominator_mul*self.layers[ind].weight.denominator.detach()
                        diff_b=self.layers[ind].bias.grad.detach()-model2.layers[ind].bias.grad.detach()
                        self.layers[ind].bias.denominator = diff_b*diff_b+denominator_mul*self.layers[ind].bias.denominator.detach()

                        self.layers[ind].weight[:] = model2.layers[ind].weight.detach()+torch.where(self.layers[ind].weight.denominator!=0.,torch.where(
                            model2.layers[ind].weight.grad!=0.,
                            torch.clip(
                                (self.layers[ind].weight.grad.detach()-model2.layers[ind].weight.grad.detach())/self.layers[ind].weight.denominator.detach(),
                                #1./(1.+step_is_fraction_of_optimizer_denominator)-1.,
                                -step_is_fraction_of_optimizer_denominator,
                                step_is_fraction_of_optimizer_denominator)\
                                *(model2.layers[ind].weight.detach()-self.layers[ind].weight.detach())*torch.abs(self.layers[ind].weight.denominator.detach()/model2.layers[ind].weight.grad.detach()),
                            0.),0.)
                        self.layers[ind].bias[:] = model2.layers[ind].bias.detach() + torch.where(self.layers[ind].bias.denominator!=0.,torch.where(
                            model2.layers[ind].bias.grad != 0.,
                            torch.clip(
                                (self.layers[ind].bias.grad.detach() - model2.layers[ind].bias.grad.detach()) /
                                self.layers[ind].bias.denominator.detach(),
                                #1. / (1. + step_is_fraction_of_optimizer_denominator) - 1.,
                                 -step_is_fraction_of_optimizer_denominator,
                                step_is_fraction_of_optimizer_denominator) \
                            * (model2.layers[ind].bias.detach() - self.layers[ind].bias.detach()) * torch.abs(
                                self.layers[ind].bias.denominator.detach() / model2.layers[ind].bias.grad.detach()),
                            0.),0.)
                        # self.layers[ind].weight[:] = torch.where(
                        #     model2.layers[ind].weight.grad*self.layers[ind].weight.grad > 0.,
                        #     model2.layers[ind].weight.detach(),
                        #     self.layers[ind].weight.detach()-torch.abs(self.layers[ind].weight.detach()-model2.layers[ind].weight.detach()))
                        # self.layers[ind].bias[:] = torch.where(
                        #     model2.layers[ind].bias.grad * self.layers[ind].bias.grad > 0.,
                        #     model2.layers[ind].bias.detach(),
                        #     self.layers[ind].bias.detach() - torch.abs(
                        #         self.layers[ind].bias.detach() - model2.layers[ind].bias.detach()))

                    else:
                        self.layers[ind].weight[:] -= torch.abs(
                            self.layers[ind].weight.detach() - model2.layers[ind].weight.detach()) * torch.sign(
                            self.layers[ind].weight.grad.detach())
                        self.layers[ind].bias[:] -= torch.abs(
                            self.layers[ind].bias.detach() - model2.layers[ind].bias.detach()) * torch.sign(
                            self.layers[ind].bias.grad.detach())

                #self.layers[ind].weight[:] =model2.layers[ind].weight.detach()
                # self.layers[ind].bias[:] = model2.layers[ind].bias.detach()
                self.layers[ind].weight.requires_grad = True
                self.layers[ind].bias.requires_grad = True

    def backward_grad_correction_with_weight_change(self,loss,model2,loss_fn,target):
        start_ind = len(self.intermediate_outputs) - 2 if loss_fn==F.cross_entropy else len(self.intermediate_outputs) - 1
        ind = start_ind#len(self.intermediate_outputs) - 1
        grad = torch.autograd.grad(outputs=loss,
                                   inputs=self.intermediate_outputs[start_ind],
                                   # inputs=self.intermediate_outputs[0],
                                   grad_outputs=None
                                   , retain_graph=True)
        self.intermediate_outputs_grad[ind] = grad[0]
        for ind in range(start_ind, -1, -1):
            if hasattr(self.layers[ind], 'weight'):
                if ind>0:
                    self.intermediate_outputs[ind] = self.layers[ind](self.intermediate_outputs[ind-1])
                else:
                    self.intermediate_outputs[ind]=self.layers[ind](self.input_cached)
                grad_outputs = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
                                                   inputs=[self.layers[ind].weight,self.layers[ind].bias],
                                                   # inputs=self.intermediate_outputs[0],
                                                   grad_outputs=self.intermediate_outputs_grad[ind]
                                                   , retain_graph=True)
                self.layers[ind].weight.grad = grad_outputs[0]
                self.layers[ind].bias.grad=grad_outputs[1]

                #self.layers[ind].weight = model2.layers[ind].weight#.clone().detach()
                #self.layers[ind].bias = model2.layers[ind].bias#.clone().detach()

                #self.layers[ind].weight.requires_grad=False
                #self.layers[ind].bias.requires_grad=False
                #self.layers[ind].weight[:]=1#torch.abs(self.layers[ind].weight-model2.layers[ind].weight)*torch.sign(self.layers[ind].weight.grad)
                #self.layers[ind].bias[:]=1#torch.abs(self.layers[ind].bias-model2.layers[ind].bias)*torch.sign(self.layers[ind].bias.grad)
                # self.layers[ind].weight.requires_grad = True
                # self.layers[ind].bias.requires_grad = True

                #model2.layers[ind].weight.grad=grad_outputs[0].detach()#.clone()
                #model2.layers[ind].bias.grad=grad_outputs[1].detach()

                # if ind > 0:
                #     grad_layers = torch.autograd.grad(outputs=self.layers[ind].weight,
                #                                       inputs=self.intermediate_outputs[ind - 1],
                #                                       # inputs=self.intermediate_outputs[0],
                #                                       grad_outputs=self.layers[ind].weight.grad
                #                                       , retain_graph=True)
                #     self.intermediate_outputs[ind - 1].grad = grad_layers[0]
            if ind > 0:
                #self.intermediate_outputs[ind]+=0.
                #self.intermediate_outputs[ind - 1]+=0.
                grad_layers = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
                                                  inputs=self.intermediate_outputs[ind - 1],
                                                  # inputs=self.intermediate_outputs[0],
                                                  grad_outputs=self.intermediate_outputs_grad[ind]
                                                  , retain_graph=True,
                                                  # allow_unused=True
                                                  )
                self.intermediate_outputs_grad[ind - 1] = grad_layers[0]

                # if ind==len(self.intermediate_outputs)-1:
                #     self.gradient_calc(self.intermediate_outputs[ind - 1],self.intermediate_outputs[ind - 1],
                #                        self.intermediate_outputs[ind],self.layers[ind])
                # pass

                #enhanced=average_gradient_of_nonlinear_layers_enhancement
                if average_gradient_of_nonlinear_layers_enhancement:
                    if isinstance(self.layers[ind], self.nonlinear_complex):
                        # self.gradient_calc(model2.intermediate_outputs[ind - 1], self.intermediate_outputs[ind - 1],
                        #                    model2.intermediate_outputs[ind],self.intermediate_outputs[ind].grad, self.layers[ind])

                        # self.gradient_calc(model2.intermediate_outputs[ind - 1], self.intermediate_outputs[ind - 1],
                        #                    self.intermediate_outputs_grad[ind - 1],
                        #                    model2.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                        #                    self.layers[ind])
                        log=False
                        if log:
                            print("----")
                            print(self.intermediate_outputs_grad[ind - 1])
                        self.gradient_calc(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                                           self.intermediate_outputs_grad[ind - 1],
                                           self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                                           self.layers[ind])
                        # self.gradient_calc_last_layer2(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                        #                    self.intermediate_outputs_grad[ind - 1],
                        #                    self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                        #                    self.layers[ind],loss_fn,target)
                        # self.intermediate_outputs_grad[ind - 1]=0.5*self.intermediate_outputs_grad[ind - 1]+\
                        #                                         model2.intermediate_outputs[ind - 1].grad * 0.5
                        # self.gradient_calc_last_layer(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                        #                    self.intermediate_outputs_grad[ind - 1],
                        #                    self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                        #                    self.layers[ind],loss_fn,target)
                        if log:
                            print(self.intermediate_outputs_grad[ind - 1])
                        #self.intermediate_outputs_grad[ind - 1]=self.intermediate_outputs_grad[ind - 1]*torch.norm(model2.intermediate_outputs_grad[ind - 1],1)/torch.norm(self.intermediate_outputs_grad[ind - 1],1)

                    elif isinstance(self.layers[ind], self.nonlinear_simple):
                        self.gradient_calc_simple(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                                           self.intermediate_outputs_grad[ind - 1],
                                           self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                                           self.layers[ind])
                        #self.intermediate_outputs_grad[ind - 1]=self.intermediate_outputs_grad[ind - 1]*torch.norm(model2.intermediate_outputs_grad[ind - 1],1)/torch.norm(self.intermediate_outputs_grad[ind - 1],1)
                #self.intermediate_outputs[ind - 1].grad.detach()

            if hasattr(self.layers[ind], 'weight'):
                #with torch.no_grad():
                self.layers[ind].weight.requires_grad=False
                self.layers[ind].bias.requires_grad=False
                self.layers[ind].weight[:]-=torch.abs(self.layers[ind].weight.detach()-model2.layers[ind].weight.detach())*torch.sign(self.layers[ind].weight.grad.detach())
                self.layers[ind].bias[:]-=torch.abs(self.layers[ind].bias.detach()-model2.layers[ind].bias.detach())*torch.sign(self.layers[ind].bias.grad.detach())
                #self.layers[ind].weight[:] =model2.layers[ind].weight.detach()
                # self.layers[ind].bias[:] = model2.layers[ind].bias.detach()
                self.layers[ind].weight.requires_grad = True
                self.layers[ind].bias.requires_grad = True

    def backward_grad_correction(self,loss,model2,loss_fn,target):
        ind = len(self.intermediate_outputs) - 1
        grad = torch.autograd.grad(outputs=loss,
                                   inputs=self.intermediate_outputs[len(self.intermediate_outputs) - 1],
                                   # inputs=self.intermediate_outputs[0],
                                   grad_outputs=None
                                   , retain_graph=True)
        self.intermediate_outputs_grad[ind] = grad[0]
        for ind in range(len(self.intermediate_outputs) - 1, -1, -1):
            if hasattr(self.layers[ind], 'weight'):
                if ind>0:
                    self.intermediate_outputs[ind] = self.layers[ind](self.intermediate_outputs[ind-1])
                else:
                    self.intermediate_outputs[ind]=self.layers[ind](self.input_cached)
                grad_outputs = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
                                                   inputs=[self.layers[ind].weight,self.layers[ind].bias],
                                                   # inputs=self.intermediate_outputs[0],
                                                   grad_outputs=self.intermediate_outputs_grad[ind]
                                                   , retain_graph=True)
                self.layers[ind].weight.grad = grad_outputs[0]
                self.layers[ind].bias.grad=grad_outputs[1]

                model2.layers[ind].weight.grad=grad_outputs[0].detach()#.clone()
                model2.layers[ind].bias.grad=grad_outputs[1].detach()

                # if ind > 0:
                #     grad_layers = torch.autograd.grad(outputs=self.layers[ind].weight,
                #                                       inputs=self.intermediate_outputs[ind - 1],
                #                                       # inputs=self.intermediate_outputs[0],
                #                                       grad_outputs=self.layers[ind].weight.grad
                #                                       , retain_graph=True)
                #     self.intermediate_outputs[ind - 1].grad = grad_layers[0]
            if ind > 0:
                grad_layers = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
                                                  inputs=self.intermediate_outputs[ind - 1],
                                                  # inputs=self.intermediate_outputs[0],
                                                  grad_outputs=self.intermediate_outputs_grad[ind]
                                                  , retain_graph=True,
                                                  #allow_unused=True
                                                  )
                self.intermediate_outputs_grad[ind - 1] = grad_layers[0]

                # if ind==len(self.intermediate_outputs)-1:
                #     self.gradient_calc(self.intermediate_outputs[ind - 1],self.intermediate_outputs[ind - 1],
                #                        self.intermediate_outputs[ind],self.layers[ind])
                # pass

                #enhanced=True
                if average_gradient_of_nonlinear_layers_enhancement:
                    if isinstance(self.layers[ind], self.nonlinear_complex):
                        # self.gradient_calc(model2.intermediate_outputs[ind - 1], self.intermediate_outputs[ind - 1],
                        #                    model2.intermediate_outputs[ind],self.intermediate_outputs[ind].grad, self.layers[ind])

                        # self.gradient_calc(model2.intermediate_outputs[ind - 1], self.intermediate_outputs[ind - 1],
                        #                    self.intermediate_outputs_grad[ind - 1],
                        #                    model2.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                        #                    self.layers[ind])
                        log=False
                        if log:
                            print("----")
                            print(self.intermediate_outputs_grad[ind - 1])
                        self.gradient_calc(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                                           self.intermediate_outputs_grad[ind - 1],
                                           self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                                           self.layers[ind])
                        # self.gradient_calc_last_layer2(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                        #                    self.intermediate_outputs_grad[ind - 1],
                        #                    self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                        #                    self.layers[ind],loss_fn,target)
                        # self.intermediate_outputs_grad[ind - 1]=0.5*self.intermediate_outputs_grad[ind - 1]+\
                        #                                         model2.intermediate_outputs[ind - 1].grad * 0.5
                        # self.gradient_calc_last_layer(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                        #                    self.intermediate_outputs_grad[ind - 1],
                        #                    self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                        #                    self.layers[ind],loss_fn,target)
                        if log:
                            print(self.intermediate_outputs_grad[ind - 1])
                        #self.intermediate_outputs_grad[ind - 1]=self.intermediate_outputs_grad[ind - 1]*torch.norm(model2.intermediate_outputs_grad[ind - 1],1)/torch.norm(self.intermediate_outputs_grad[ind - 1],1)

                    elif isinstance(self.layers[ind], self.nonlinear_simple):
                        self.gradient_calc_simple(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                                           self.intermediate_outputs_grad[ind - 1],
                                           self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                                           self.layers[ind])
                        #self.intermediate_outputs_grad[ind - 1]=self.intermediate_outputs_grad[ind - 1]*torch.norm(model2.intermediate_outputs_grad[ind - 1],1)/torch.norm(self.intermediate_outputs_grad[ind - 1],1)
                #self.intermediate_outputs[ind - 1].grad.detach()

    def backward(self,loss):
        start_ind = len(self.intermediate_outputs) - 1
        #start_ind=len(self.intermediate_outputs)-2 if
        ind=start_ind
        grad = torch.autograd.grad(outputs=loss,
                                   inputs=self.intermediate_outputs[start_ind],
                                   # inputs=self.intermediate_outputs[0],
                                   grad_outputs=None
                                   , retain_graph=True)
        self.intermediate_outputs[ind].grad=grad[0]
        if average_gradient_of_loss:
            self.intermediate_outputs_grad[ind]=grad[0]
        # if average_gradient_of_linear_layers_enhancement:
        #     self.intermediate_outputs_grad[ind]=grad[0]
        for ind in range(start_ind,-1,-1):
            if hasattr(self.layers[ind],'weight'):
                grad_outputs = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
                                           inputs=(self.layers[ind].weight,self.layers[ind].bias),
                                           # inputs=self.intermediate_outputs[0],
                                           grad_outputs=self.intermediate_outputs[ind].grad
                                           , retain_graph=True)
                self.layers[ind].weight.grad=grad_outputs[0]
                self.layers[ind].bias.grad = grad_outputs[1]
                # if ind > 0:
                #     grad_layers = torch.autograd.grad(outputs=self.layers[ind].weight,
                #                                       inputs=self.intermediate_outputs[ind - 1],
                #                                       # inputs=self.intermediate_outputs[0],
                #                                       grad_outputs=self.layers[ind].weight.grad
                #                                       , retain_graph=True)
                #     self.intermediate_outputs[ind - 1].grad = grad_layers[0]
            if ind > 0:
                grad_layers = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
                                                  inputs=self.intermediate_outputs[ind - 1],
                                                  # inputs=self.intermediate_outputs[0],
                                                  grad_outputs=self.intermediate_outputs[ind].grad
                                                  , retain_graph=True)
                self.intermediate_outputs[ind - 1].grad = grad_layers[0]
                # if average_gradient_of_linear_layers_enhancement:
                #     self.intermediate_outputs_grad[ind-1] = grad_layers[0]
                # if ind==len(self.intermediate_outputs)-1:
                #     self.gradient_calc(self.intermediate_outputs[ind - 1],self.intermediate_outputs[ind - 1],
                #                        self.intermediate_outputs[ind],self.layers[ind])
                    #pass
                # if isinstance(self.layers[ind],self.nonlinear_complex):
                #     self.gradient_calc(self.intermediate_outputs[ind - 1], self.intermediate_outputs[ind - 1],
                #                        self.intermediate_outputs[ind], self.layers[ind])
                #pass

    # def train(self, mode: bool = True):
    #     super().train(mode)
    def forward(self, x):
        self.input_cached=x
        #layers_num=len(self.layers)
        #for i in range(layers_num-1):
        #    x=self.layers[i](x)
        #output=self.layers[layers_num-1](x)
        #return output

        i=0
        for layer in self.layers:
            if self.intermediate_outputs[i] is None:
                x=layer(x)
                #self.intermediate_outputs[i]=(Variable(x, requires_grad=True))
                self.intermediate_outputs[i]=x
                self.intermediate_outputs[i].require_grad=True

                #self.intermediate_outputs_grad[i]=torch.empty_like(x)
            else:
                self.intermediate_outputs[i] = layer(x)
                x=self.intermediate_outputs[i]
            i+=1
            if i==len(self.layers)-1 and isinstance(self.layers[len(self.layers)-1],torch.nn.Softmax) and self.training:
                break
        return x

    @torch.no_grad()
    def copy_grad_from(self,model):
        for ind in range(0,len(self.layers)):
            if hasattr(self.layers[ind],'weight') and hasattr(model.layers[ind],'weight') and model.layers[ind].weight is not None:
                if hasattr(self.layers[ind].weight,'grad') and self.layers[ind].weight.grad is not None:
                    self.layers[ind].weight.grad[:]=model.layers[ind].weight.grad[:]
                else:
                    self.layers[ind].weight.grad = model.layers[ind].weight.grad.detach().clone()
                    #self.layers[ind].weight.grad = model.layers[ind].weight.grad.detach()
            if hasattr(self.layers[ind],'bias') and hasattr(model.layers[ind],'bias') and model.layers[ind].bias is not None:
                if hasattr(self.layers[ind].bias,'grad') and self.layers[ind].bias.grad is not None:
                    self.layers[ind].bias.grad[:]=model.layers[ind].bias.grad[:]
                else:
                    # if model.layers[ind].bias is None:
                    #     print('aa')
                    self.layers[ind].bias.grad = model.layers[ind].bias.grad.detach().clone()
                    #self.layers[ind].bias.grad = model.layers[ind].bias.grad.detach()

    @torch.no_grad()
    def add_grad_from(self, model):
        for ind in range(0, len(self.layers)):
            if hasattr(self.layers[ind], 'weight') and hasattr(model.layers[ind], 'weight') and model.layers[
                ind].weight is not None:
                if hasattr(self.layers[ind].weight, 'grad') and self.layers[ind].weight.grad is not None:
                    self.layers[ind].weight.grad[:] += model.layers[ind].weight.grad[:]
                else:
                    self.layers[ind].weight.grad = model.layers[ind].weight.grad.detach().clone()
                    # self.layers[ind].weight.grad = model.layers[ind].weight.grad.detach()
            if hasattr(self.layers[ind], 'bias') and hasattr(model.layers[ind], 'bias') and model.layers[
                ind].bias is not None:
                if hasattr(self.layers[ind].bias, 'grad') and self.layers[ind].bias.grad is not None:
                    self.layers[ind].bias.grad[:] += model.layers[ind].bias.grad[:]
                else:
                    # if model.layers[ind].bias is None:
                    #     print('aa')
                    self.layers[ind].bias.grad = model.layers[ind].bias.grad.detach().clone()
                    # self.layers[ind].bias.grad = model.layers[ind].bias.grad.detach()
    def erase_grad(self):
        for ind in range(0,len(self.layers)):
            if hasattr(self.layers[ind],'weight') and hasattr(self.layers[ind].weight,'grad') and self.layers[ind].weight.grad is not None:
                del self.layers[ind].weight.grad
            if hasattr(self.layers[ind],'bias') and hasattr(self.layers[ind].bias,'grad') and self.layers[ind].bias.grad is not None:
                del self.layers[ind].bias.grad
    @torch.no_grad()
    def distance_from(self,model,l=1):
        d=float(0)
        for ind in range(0,len(self.layers)):
            if hasattr(self.layers[ind],'weight') and hasattr(model.layers[ind],'weight'):
                d+=(self.layers[ind].weight[:]-model.layers[ind].weight[:]).abs().pow(l).sum()

            if hasattr(self.layers[ind],'bias') and hasattr(model.layers[ind],'bias') and self.layers[ind].bias is not None:
                d+=(self.layers[ind].bias[:]-model.layers[ind].bias[:]).abs().pow(l).sum()
        d=d**(1./l)
        return d

    def backward_grad_correction_with_weight_change2_parametrized_relative_step_size_and_no_negations_in_step_direction(self,loss,model2,weight_change=True,accumulate_gradients=False,lr_mul=1):
        #start_ind = len(self.intermediate_outputs) - 2 if loss_fn==F.cross_entropy else len(self.intermediate_outputs) - 1
        start_ind = len(self.intermediate_outputs) - 1
        ind = start_ind#len(self.intermediate_outputs) - 1
        grad = torch.autograd.grad(outputs=loss,
                                   inputs=self.intermediate_outputs[start_ind],
                                   # inputs=self.intermediate_outputs[0],
                                   grad_outputs=None
                                   , retain_graph=True)
        self.intermediate_outputs_grad[ind] = grad[0]

        if average_gradient_of_loss:
            self.intermediate_outputs_grad[ind]=0.5*(grad[0]+model2.intermediate_outputs_grad[ind])


        for ind in range(start_ind, -1, -1):
            if hasattr(self.layers[ind], 'weight'):


                if ind>0:
                    self.intermediate_outputs[ind] = self.layers[ind](self.intermediate_outputs[ind-1])
                else:
                    self.intermediate_outputs[ind]=self.layers[ind](self.input_cached)

                grad_outputs = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
                                                   inputs=[self.layers[ind].weight,self.layers[ind].bias],
                                                   # inputs=self.intermediate_outputs[0],
                                                   grad_outputs=self.intermediate_outputs_grad[ind]
                                                   , retain_graph=True)
                if not average_gradient_of_linear_layers_enhancement:
                    if accumulate_gradients and self.layers[ind].weight.grad is not None:
                        self.layers[ind].weight.grad += grad_outputs[0]
                        self.layers[ind].bias.grad+=grad_outputs[1]
                    else:
                        self.layers[ind].weight.grad = grad_outputs[0]
                        self.layers[ind].bias.grad = grad_outputs[1]
                else:
                    grad_outputs2 = torch.autograd.grad(outputs=model2.intermediate_outputs[ind],
                                                       inputs=(model2.layers[ind].weight, model2.layers[ind].bias),
                                                       # inputs=self.intermediate_outputs[0],
                                                       grad_outputs=self.intermediate_outputs_grad[ind]
                                                       , retain_graph=True)
                    if accumulate_gradients:
                        self.layers[ind].weight.grad += 0.5*(grad_outputs[0]+grad_outputs2[0])
                        self.layers[ind].bias.grad+=0.5*(grad_outputs[1]+grad_outputs2[1])
                    else:
                        self.layers[ind].weight.grad = 0.5 * (grad_outputs[0] + grad_outputs2[0])
                        self.layers[ind].bias.grad = 0.5 * (grad_outputs[1] + grad_outputs2[1])


            if ind > 0:
                #self.intermediate_outputs[ind]+=0.
                #self.intermediate_outputs[ind - 1]+=0.


                grad_layers = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
                                                  inputs=self.intermediate_outputs[ind - 1],
                                                  # inputs=self.intermediate_outputs[0],
                                                  grad_outputs=self.intermediate_outputs_grad[ind]
                                                  , retain_graph=True,
                                                  # allow_unused=True
                                                  )
                self.intermediate_outputs_grad[ind - 1] = grad_layers[0]

                if average_gradient_of_linear_layers_enhancement and not (average_gradient_of_nonlinear_layers_enhancement and isinstance(self.layers[ind], self.nonlinear_simple)):
                    grad_layers2 = torch.autograd.grad(outputs=model2.intermediate_outputs[ind],
                                                      inputs=model2.intermediate_outputs[ind - 1],
                                                      # inputs=self.intermediate_outputs[0],
                                                      grad_outputs=self.intermediate_outputs_grad[ind]
                                                      , retain_graph=True)
                    self.intermediate_outputs_grad[ind - 1] = 0.5*(self.intermediate_outputs_grad[ind - 1]+grad_layers2[0])

                # if ind==len(self.intermediate_outputs)-1:
                #     self.gradient_calc(self.intermediate_outputs[ind - 1],self.intermediate_outputs[ind - 1],
                #                        self.intermediate_outputs[ind],self.layers[ind])
                # pass

                #enhanced=True
                if average_gradient_of_nonlinear_layers_enhancement:

                    if isinstance(self.layers[ind], self.nonlinear_complex):#deprecated
                        # self.gradient_calc(model2.intermediate_outputs[ind - 1], self.intermediate_outputs[ind - 1],
                        #                    model2.intermediate_outputs[ind],self.intermediate_outputs[ind].grad, self.layers[ind])

                        # self.gradient_calc(model2.intermediate_outputs[ind - 1], self.intermediate_outputs[ind - 1],
                        #                    self.intermediate_outputs_grad[ind - 1],
                        #                    model2.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                        #                    self.layers[ind])
                        log=False
                        if log:
                            print("----")
                            print(self.intermediate_outputs_grad[ind - 1])
                        self.gradient_calc(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                                           self.intermediate_outputs_grad[ind - 1],
                                           self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                                           self.layers[ind])

                        if log:
                            print(self.intermediate_outputs_grad[ind - 1])
                        #self.intermediate_outputs_grad[ind - 1]=self.intermediate_outputs_grad[ind - 1]*torch.norm(model2.intermediate_outputs_grad[ind - 1],1)/torch.norm(self.intermediate_outputs_grad[ind - 1],1)

                    elif isinstance(self.layers[ind], self.nonlinear_simple):
                        self.gradient_calc_simple(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                                           self.intermediate_outputs_grad[ind - 1],
                                           self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                                           self.layers[ind])
                        #self.intermediate_outputs_grad[ind - 1]=self.intermediate_outputs_grad[ind - 1]*torch.norm(model2.intermediate_outputs_grad[ind - 1],1)/torch.norm(self.intermediate_outputs_grad[ind - 1],1)

                #self.intermediate_outputs[ind - 1].grad.detach()

            if weight_change and hasattr(self.layers[ind], 'weight'):
                self.layers[ind].weight.requires_grad=False
                self.layers[ind].bias.requires_grad=False

                # if step_is_fraction_of_optimizer_denominator!=0.:
                #
                #     diff_w=self.layers[ind].weight.grad.detach()-model2.layers[ind].weight.grad.detach()
                #     self.layers[ind].weight.denominator=diff_w*diff_w+denominator_mul*self.layers[ind].weight.denominator.detach()
                #     diff_b=self.layers[ind].bias.grad.detach()-model2.layers[ind].bias.grad.detach()
                #     self.layers[ind].bias.denominator = diff_b*diff_b+denominator_mul*self.layers[ind].bias.denominator.detach()
                #
                #     self.layers[ind].weight[:] = model2.layers[ind].weight.detach()+torch.where(self.layers[ind].weight.denominator!=0.,torch.where(
                #         model2.layers[ind].weight.grad!=0.,
                #         torch.clip(
                #             (self.layers[ind].weight.grad.detach()-model2.layers[ind].weight.grad.detach())/self.layers[ind].weight.denominator.detach(),
                #             #1./(1.+step_is_fraction_of_optimizer_denominator)-1.,
                #             -step_is_fraction_of_optimizer_denominator,
                #             step_is_fraction_of_optimizer_denominator)\
                #             *(model2.layers[ind].weight.detach()-self.layers[ind].weight.detach())*torch.abs(self.layers[ind].weight.denominator.detach()/model2.layers[ind].weight.grad.detach()),
                #         0.),0.)
                #     self.layers[ind].bias[:] = model2.layers[ind].bias.detach() + torch.where(self.layers[ind].bias.denominator!=0.,torch.where(
                #         model2.layers[ind].bias.grad != 0.,
                #         torch.clip(
                #             (self.layers[ind].bias.grad.detach() - model2.layers[ind].bias.grad.detach()) /
                #             self.layers[ind].bias.denominator.detach(),
                #             #1. / (1. + step_is_fraction_of_optimizer_denominator) - 1.,
                #              -step_is_fraction_of_optimizer_denominator,
                #             step_is_fraction_of_optimizer_denominator) \
                #         * (model2.layers[ind].bias.detach() - self.layers[ind].bias.detach()) * torch.abs(
                #             self.layers[ind].bias.denominator.detach() / model2.layers[ind].bias.grad.detach()),
                #         0.),0.)
                #
                #
                # else:
                #     self.layers[ind].weight[:] -= torch.abs(
                #         self.layers[ind].weight.detach() - model2.layers[ind].weight.detach()) * torch.sign(
                #         self.layers[ind].weight.grad.detach())
                #     self.layers[ind].bias[:] -= torch.abs(
                #         self.layers[ind].bias.detach() - model2.layers[ind].bias.detach()) * torch.sign(
                #         self.layers[ind].bias.grad.detach())
                if method==6:
                    self.layers[ind].weight[:] -= torch.abs(
                        self.layers[ind].weight.detach() - model2.layers[ind].weight.detach()) * (torch.sign(
                        self.layers[ind].weight.grad.detach())+0.9*torch.sign(
                        model2.layers[ind].weight.grad.detach()))*(lr_mul/1.9)
                    self.layers[ind].bias[:] -= torch.abs(
                        self.layers[ind].bias.detach() - model2.layers[ind].bias.detach()) * (torch.sign(
                        self.layers[ind].bias.grad.detach())+0.9*torch.sign(
                        model2.layers[ind].bias.grad.detach()))*(lr_mul/1.9)
                else:
                    self.layers[ind].weight[:] -= torch.abs(
                        self.layers[ind].weight.detach() - model2.layers[ind].weight.detach()) * (torch.sign(
                        self.layers[ind].weight.grad.detach()) + torch.sign(
                        model2.layers[ind].weight.grad.detach())) * (0.5 * lr_mul)
                    self.layers[ind].bias[:] -= torch.abs(
                        self.layers[ind].bias.detach() - model2.layers[ind].bias.detach()) * (torch.sign(
                        self.layers[ind].bias.grad.detach()) + torch.sign(
                        model2.layers[ind].bias.grad.detach())) * (0.5 * lr_mul)

                    # print(torch.where(torch.sign(
                    #     self.layers[ind].weight.grad.detach()) * torch.sign(
                    #     model2.layers[ind].weight.grad.detach()) == -1.))

                #self.layers[ind].weight[:] =model2.layers[ind].weight.detach()
                # self.layers[ind].bias[:] = model2.layers[ind].bias.detach()
                self.layers[ind].weight.requires_grad = True
                self.layers[ind].bias.requires_grad = True
    def set_require_grad(self,require_grad=True):
        pass

    @torch.no_grad
    def soft_copy_grad(self,model,alpha=0.5):
        for ind in range(0,len(self.layers)):
            if hasattr(self.layers[ind],'weight') and hasattr(model.layers[ind],'weight') and model.layers[ind].weight is not None:
                if hasattr(self.layers[ind].weight,'grad') and self.layers[ind].weight.grad is not None:
                    self.layers[ind].weight.grad[:]=self.layers[ind].weight.grad*(1.-alpha)+alpha*model.layers[ind].weight.grad[:]
                else:
                    self.layers[ind].weight.grad = model.layers[ind].weight.grad.detach().clone()
                    #self.layers[ind].weight.grad = model.layers[ind].weight.grad.detach()
            if hasattr(self.layers[ind],'bias') and hasattr(model.layers[ind],'bias') and model.layers[ind].bias is not None:
                if hasattr(self.layers[ind].bias,'grad') and self.layers[ind].bias.grad is not None:
                    self.layers[ind].bias.grad[:]=self.layers[ind].bias.grad*(1.-alpha)+alpha*model.layers[ind].bias.grad[:]
                else:
                    self.layers[ind].bias.grad = model.layers[ind].bias.grad.detach().clone()
                    #self.layers[ind].bias.grad = model.layers[ind].bias.grad.detach()

    @torch.no_grad
    def avg_grad_to_integrated_grad(self, model_updated, model):
        for ind in range(0, len(self.layers)):
            if hasattr(self.layers[ind], 'weight') and hasattr(self.layers[ind], 'weight') and self.layers[
                ind].weight is not None:
                if hasattr(self.layers[ind].weight, 'grad') and self.layers[ind].weight.grad is not None:
                    self.layers[ind].weight.grad[:] = self.layers[ind].weight.grad * (model_updated.layers[ind].weight-model.layers[ind].weight)
                # else:
                #     self.layers[ind].weight.grad = self.layers[ind].weight.grad.detach().clone()
                #     # self.layers[ind].weight.grad = self.layers[ind].weight.grad.detach()
            if hasattr(self.layers[ind], 'bias') and hasattr(self.layers[ind], 'bias') and self.layers[
                ind].bias is not None:
                if hasattr(self.layers[ind].bias, 'grad') and self.layers[ind].bias.grad is not None:
                    self.layers[ind].bias.grad[:] = self.layers[ind].bias.grad * (model_updated.layers[ind].bias-model.layers[ind].bias)
                # else:
                #     self.layers[ind].bias.grad = self.layers[ind].bias.grad.detach().clone()
                #     # self.layers[ind].bias.grad = self.layers[ind].bias.grad.detach()
class NN_Advanced(NN):
    param_to_optimizer_param_mapping_gradient_magnitudes={}
    def __init__(self,layers):
        #super(NN,self).__init__(layers)
        NN.__init__(self,layers)
        # self.intermediate_outputs2=[None for i in range(len(self.layers))]
    def init_param_to_optimizer_param_mapping(self,optimizer):
        if self.param_to_optimizer_param_mapping_gradient_magnitudes:
            opt_values = list(optimizer.state.values())
            key=''
            if type(optimizer) == torch.optim.Adam:
                key='exp_avg_sq'
            elif type(optimizer)==torch.optim.RMSprop:
                key='square_avg'
            for i in range(len(opt_values)):
                self.param_to_optimizer_param_mapping_gradient_magnitudes[opt_values[i]]=opt_values[i][key]

    mask=torch.arange(0)
    #mask2 = torch.arange(0)
    # def initial_param_to_optimizer_param_mapping(self,params,optimizer):
    #     return torch.ones(params.shape())*optimizer.param_groups[0]['lr']
    def initial_param_to_optimizer_param_mapping(self,params,optimizer):
        return torch.ones(params.shape())*optimizer.param_groups[0]['lr']
    def prepare_param_translate_mask(self,params):
        params_flat_view = params.view((-1))
        params_flat_view_len = len(params_flat_view)
        # params_flat_view_len+=params_flat_view_len%2
        if len(self.mask.view((-1))) < params_flat_view_len:
            self.mask = torch.zeros((params_flat_view_len,), dtype=d_type)
            for ind in range(len(self.mask)):
                if ind%2==0:
                    self.mask[ind]=-1
                else:
                    self.mask[ind]=1
                #self.mask[ind] = (ind % 2) * 2 - 1
            #print(self.mask)
        return params_flat_view,params_flat_view_len
    def translate_params(self,step_coefficient,optimizer=None):#STEP COEFFICIENT PROPERTY IS CACHED IN prepare_param_translate_mask

        with torch.no_grad():
            mapping_param_to_optimizer_param = bool(self.param_to_optimizer_param_mapping_gradient_magnitudes)
            if mapping_param_to_optimizer_param:
                for key in self.param_to_optimizer_param_mapping_gradient_magnitudes.keys():
                    key+=step_coefficient*self.param_to_optimizer_param_mapping_gradient_magnitudes[key]
            else:
                lr=optimizer.param_groups[0]['lr']
                mul=lr*step_coefficient
                for ind in range(0, len(self.layers)):
                    if hasattr(self.layers[ind], 'weight'):
                        weight_view,weight_count=self.prepare_param_translate_mask(self.layers[ind].weight)
                        bias_view,bias_count=self.prepare_param_translate_mask(self.layers[ind].bias)
                        weight_view+=mul*self.mask[0:weight_count]*weight_view
                        bias_view +=mul * self.mask[0:bias_count]*bias_view
                        #self.layers[ind].weight+=step_coefficient*lr*self.layers[ind].weight
                        #self.layers[ind].bias+=step_coefficient*lr*self.layers[ind].bias

    # def forward2(self, x):
    #     self.input_cached2=x
    #     #layers_num=len(self.layers)
    #     #for i in range(layers_num-1):
    #     #    x=self.layers[i](x)
    #     #output=self.layers[layers_num-1](x)
    #     #return output
    #
    #     i=0
    #     for layer in self.layers:
    #         if self.intermediate_outputs2[i] is None:
    #             x=layer(x)
    #             #self.intermediate_outputs[i]=(Variable(x, requires_grad=True))
    #             self.intermediate_outputs2[i]=x
    #             self.intermediate_outputs2[i].require_grad=True
    #
    #             #self.intermediate_outputs_grad[i]=torch.empty_like(x)
    #         else:
    #             self.intermediate_outputs2[i] = layer(x)
    #             x=self.intermediate_outputs2[i]
    #         i+=1
    #         if i==len(self.layers)-1 and isinstance(self.layers[len(self.layers)-1],torch.nn.Softmax) and self.training:
    #             break
    #     return x
    def backward_range_grad(self, loss, loss2,model2,accumulate_gradients=False):
        # start_ind = len(self.intermediate_outputs) - 2 if loss_fn==F.cross_entropy else len(self.intermediate_outputs) - 1
        start_ind = len(self.intermediate_outputs) - 1
        ind = start_ind  # len(self.intermediate_outputs) - 1
        grad = torch.autograd.grad(outputs=loss,
                                   inputs=self.intermediate_outputs[start_ind],
                                   # inputs=self.intermediate_outputs[0],
                                   grad_outputs=None
                                   , retain_graph=True)
        #self.intermediate_outputs_grad[ind] = grad[0]

        if average_gradient_of_loss:
            grad2 = torch.autograd.grad(outputs=loss2,
                                       inputs=model2.intermediate_outputs[start_ind],
                                       # inputs=self.intermediate_outputs[0],
                                       grad_outputs=None
                                       , retain_graph=True)
            self.intermediate_outputs_grad[ind] = 0.5 * (grad[0] + grad2[0])
        else:
            self.intermediate_outputs_grad[ind] = grad[0]


        for ind in range(start_ind, -1, -1):
            if hasattr(self.layers[ind], 'weight'):


                grad_outputs = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
                                                   inputs=[self.layers[ind].weight, self.layers[ind].bias],
                                                   # inputs=self.intermediate_outputs[0],
                                                   grad_outputs=self.intermediate_outputs_grad[ind]
                                                   , retain_graph=True)
                if not average_gradient_of_linear_layers_enhancement:
                    if accumulate_gradients and self.layers[ind].weight.grad is not None:
                        self.layers[ind].weight.grad += grad_outputs[0]
                        self.layers[ind].bias.grad += grad_outputs[1]
                    else:
                        self.layers[ind].weight.grad = grad_outputs[0]
                        self.layers[ind].bias.grad = grad_outputs[1]
                else:
                    grad_outputs2 = torch.autograd.grad(outputs=model2.intermediate_outputs[ind],
                                                        inputs=(model2.layers[ind].weight, model2.layers[ind].bias),
                                                        # inputs=self.intermediate_outputs[0],
                                                        grad_outputs=self.intermediate_outputs_grad[ind]
                                                        , retain_graph=True)
                    if accumulate_gradients:
                        self.layers[ind].weight.grad += 0.5 * (grad_outputs[0] + grad_outputs2[0])
                        self.layers[ind].bias.grad += 0.5 * (grad_outputs[1] + grad_outputs2[1])
                    else:
                        self.layers[ind].weight.grad = 0.5 * (grad_outputs[0] + grad_outputs2[0])
                        self.layers[ind].bias.grad = 0.5 * (grad_outputs[1] + grad_outputs2[1])

            if ind > 0:
                # self.intermediate_outputs[ind]+=0.
                # self.intermediate_outputs[ind - 1]+=0.

                grad_layers = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
                                                  inputs=self.intermediate_outputs[ind - 1],
                                                  # inputs=self.intermediate_outputs[0],
                                                  grad_outputs=self.intermediate_outputs_grad[ind]
                                                  , retain_graph=True,
                                                  # allow_unused=True
                                                  )
                self.intermediate_outputs_grad[ind - 1] = grad_layers[0]

                if average_gradient_of_linear_layers_enhancement:
                    grad_layers2 = torch.autograd.grad(outputs=model2.intermediate_outputs[ind],
                                                       inputs=model2.intermediate_outputs[ind - 1],
                                                       # inputs=self.intermediate_outputs[0],
                                                       grad_outputs=self.intermediate_outputs_grad[ind]
                                                       , retain_graph=True)
                    self.intermediate_outputs_grad[ind - 1] = 0.5 * (
                                self.intermediate_outputs_grad[ind - 1] + grad_layers2[0])

                # if ind==len(self.intermediate_outputs)-1:
                #     self.gradient_calc(self.intermediate_outputs[ind - 1],self.intermediate_outputs[ind - 1],
                #                        self.intermediate_outputs[ind],self.layers[ind])
                # pass

                # enhanced=True
                if average_gradient_of_nonlinear_layers_enhancement:

                    if isinstance(self.layers[ind], self.nonlinear_complex):  # deprecated

                        log = False
                        if log:
                            print("----")
                            print(self.intermediate_outputs_grad[ind - 1])
                        self.gradient_calc(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                                           self.intermediate_outputs_grad[ind - 1],
                                           self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                                           self.layers[ind])

                        if log:
                            print(self.intermediate_outputs_grad[ind - 1])
                        # self.intermediate_outputs_grad[ind - 1]=self.intermediate_outputs_grad[ind - 1]*torch.norm(model2.intermediate_outputs_grad[ind - 1],1)/torch.norm(self.intermediate_outputs_grad[ind - 1],1)

                    elif isinstance(self.layers[ind], self.nonlinear_simple):
                        self.gradient_calc_simple(self.intermediate_outputs[ind - 1],
                                                  model2.intermediate_outputs[ind - 1],
                                                  self.intermediate_outputs_grad[ind - 1],
                                                  self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                                                  self.layers[ind])

class DependencyGraphNode(object):
    # previous=set()
    # next=set()
    # module=None
    #
    # input_var=None
    # output_var=None#if there is no residual connection, then output_var=next_node.input_var. Else if residual connection ends just after the node, then output_var!=next_node.input_var
    input_var=None
    output_var=None
    def init(self,module):
        self.name=module.__class__.__name__
        self.previous=set()
        self.next=set()
        self.module=module

        self.input_var=None
        self.output_var=None#if there is no residual connection, then output_var=next_node.input_var. Else if residual connection ends just after the node, then output_var!=next_node.input_var

    def __init__(self,module):
        self.init(module)
        #self.module=module
    def __init__(self,module,previous:[]):
        self.init(module)
        if previous:
            for i in range(len(previous)-1,-1,-1):
                if previous[i] is None:
                    del previous[i]
            self.previous.update(previous)
        self.synchronize()
        #self.module=module

    def synchronize(self):
        for node in self.previous:
            node.next.add(self)
        for node in self.next:
            node.previous.add(self)
    def first(self):
        node=self
        while node.previous:
            node=next(iter(node.previous))
        return node
    def last(self):
        node=self
        while node.next:
            node=next(iter(node.next))
        return node
    def find_all_nodes(self):
        first_node=self.first()
        all=set()
        all.add(first_node)
        self._find_all_next_nodes(all)
        return all

    def _find_all_next_nodes(self,actual_set=set()):
        if self.next.issubset(actual_set):
            return
        actual_set.update(self.next)
        for node in self.next:
            node._find_all_next_nodes(actual_set)

    def find_all_vars(self):
        all_nodes=self.find_all_nodes()
        all_vars=set()
        for node in all_nodes:
            if node.input_var is not None:
                all_vars.add(node.input_var)
            if node.output_var is not None:
                all_vars.add(node.output_var)
        return all_vars

    # def execute(self,input):
    #     self.output_var=self.module(input)
    def execute(self):
        if len(self.previous)==1 and len(next(iter(self.previous)).next)==1:
            self.input_var=next(iter(self.previous)).output_var
        elif len(self.previous)>0:# or len(self.previous)>1:
            #prev_output_first=next(iter(self.previous)).output_var

            # if self.input_var is None or self.input_var.shape!=prev_output_first.shape:
            #     self.input_var=torch.zeros_like(prev_output_first)
            #     #self.input_var.requires_grad=True
            # else:
            #     self.input_var.zero_()
            # for prev_node in self.previous:
            #     self.input_var+=prev_node.output_var

            prev_node_iter=iter(self.previous)
            self.input_var = next(prev_node_iter).output_var.clone()
            for node in prev_node_iter:
                self.input_var+=node.output_var
        # else:
        #     assert self.input_var
        self.output_var = self.module(self.input_var)
class DependencyGraph(object):
    nonlinear_simple = (torch.nn.GELU,torch.nn.ELU, torch.nn.Sigmoid, torch.nn.Tanh,torch.nn.ReLU,torch.nn.SiLU)#(torch.nn.SELU, torch.nn.ELU, torch.nn.Sigmoid, torch.nn.Tanh,torch.nn.ReLU)#, torch.nn.BatchNorm2d)
    first_layer = None
    last_layer = None
    node_set=set()
    nodes_ordered=[]
    _variable_set=set()
    require_grad=True
    def __init__(self,layers):
        if isinstance(layers,list):
            next_node = None
            for layer in layers:
                next_node = DependencyGraphNode(layer, [next_node])
            self.init_from_node(next_node)
        elif isinstance(layers,DependencyGraphNode):
            self.init_from_node(layers)
    # def __init__(self,layers:[]):
    #     #super(NN,self).__init__(layers)
    #     #nodes=[]
    #     next_node=None
    #     for layer in layers:
    #         next_node=DependencyGraphNode(layer,next_node)
    #
    #     self.init(next_node)
    #     # self.last_layer=next_node
    #     # self.first_layer=next_node.first()
    #     #
    #     # self.node_set=self.first_layer.find_all_nodes()
    #     # self.order_nodes()
    #     #NN.__init__(self,layers)
    # def __init__(self,node:DependencyGraphNode):
    #     self.init(node)
    #     # self.first_layer=node.first()
    #     # self.last_layer=node.last()
    #     #
    #     # self.node_set=self.first_layer.find_all_nodes()
    #     # self.order_nodes()
    #     #NN.__init__(self, list(node_set))
    def init_from_node(self,node:DependencyGraphNode):
        self.last_layer = node.last()
        self.first_layer = node.first()

        self.node_set = self.first_layer.find_all_nodes()
        self.order_nodes()

        self._variable_set=self.first_layer.find_all_vars()
    def get_variable_set(self):
        self._variable_set = self.first_layer.find_all_vars()
        return self._variable_set
    # def order_nodes(self):
    #     self.nodes_ordered=[]
    #     executed_nodes=set()
    #     not_executed_nodes=set(self.node_set)
    #     iter_count=len(not_executed_nodes)
    #     for _ in range(iter_count):
    #         for node in not_executed_nodes:
    #             if executed_nodes.issuperset(node.previous):
    #                 executed_nodes.add(node)
    #                 self.nodes_ordered.append(node)
    #                 not_executed_nodes.remove(node)
    #                 break
    #     assert len(not_executed_nodes)==0
    #     assert len(self.nodes_ordered)==len(self.node_set)
    def order_nodes(self):
        self.nodes_ordered=[]
        executed_nodes=set()
        not_executed_nodes=set(self.node_set)
        iter_count=len(not_executed_nodes)
        for _ in range(iter_count):
            candidate_nodes=[]
            for node in not_executed_nodes:
                if executed_nodes.issuperset(node.previous):
                    #executed_nodes.add(node)
                    candidate_nodes.append(node)
                    #self.nodes_ordered.append(node)
                    #not_executed_nodes.remove(node)
                    #break
            executed_nodes.update(candidate_nodes)
            not_executed_nodes=not_executed_nodes-set(candidate_nodes)
            candidate_nodes.sort(key=lambda node: node.name)
            self.nodes_ordered+=candidate_nodes
        assert len(not_executed_nodes)==0
        assert len(self.nodes_ordered)==len(self.node_set)
    def execute_graph(self,input):
        self.first_layer.input_var=input
        self.first_layer.input_var.requires_grad=self.require_grad
        # if self.first_layer.output_var is None:
        #     #self.first_layer.input_var.requires_grad=True
        #     for node in self.nodes_ordered:
        #         #node.input_var.requires_grad=True
        #         node.execute()
        #         #node.output_var.requires_grad=True
        #     # self.variable_set=self.first_layer.find_all_vars()
        #     # for variable in self.variable_set:
        #     #     variable.requires_grad=True
        # else:
        #     for node in self.nodes_ordered:
        #         node.execute()
        for node in self.nodes_ordered:
            node.execute()

        if not self.require_grad:
            for node in self.nodes_ordered:
                del node.input_var
                if not node is self.last_layer:
                    del node.output_var

        return self.last_layer.output_var
    def backward_avg_grad(self,loss,dependency_graph2,weight_change=True,accumulate_gradients=False,gradient_only_modification=False):
        #start_ind = len(self.intermediate_outputs) - 2 if loss_fn==F.cross_entropy else len(self.intermediate_outputs) - 1
        start_ind = len(self.nodes_ordered) - 1#it is assumed that the last node is not softmax
        ind = start_ind#len(self.intermediate_outputs) - 1
        grad = torch.autograd.grad(outputs=loss,
                                   inputs=self.nodes_ordered[start_ind].output_var,
                                   # inputs=self.intermediate_outputs[0],
                                   grad_outputs=None
                                   , retain_graph=True)
        self.nodes_ordered[ind].output_var.grad = grad[0]

        if average_gradient_of_loss:
            self.nodes_ordered[ind].output_var.grad=0.5*(grad[0]+dependency_graph2.nodes_ordered[ind].output_var.grad)


        for ind in range(start_ind, -1, -1):
            # self.nodes_ordered[ind].execute()
            # if ind==start_ind:
            #     self.nodes_ordered[ind].output_var.grad = grad[0]
            #
            #     if average_gradient_of_loss:
            #         self.nodes_ordered[ind].output_var.grad = 0.5 * (
            #                     grad[0] + dependency_graph2.nodes_ordered[ind].output_var.grad)

            if len(self.nodes_ordered[ind].next)==1:
                # if not self.nodes_ordered[ind].output_var is next(iter(self.nodes_ordered[ind].next)).input_var:
                #     #print(self.nodes_ordered[ind].output_var)
                #     #print(next(iter(self.nodes_ordered[ind].next)).input_var)
                #     print(self.nodes_ordered[ind].name)
                if next(iter(self.nodes_ordered[ind].next)).input_var is not self.nodes_ordered[ind].output_var:
                    #self.nodes_ordered[ind].output_var.grad=next(iter(self.nodes_ordered[ind].next)).input_var.grad.clone().detach()
                    #print('aaa')
                    with torch.no_grad():
                        if len(next(iter(self.nodes_ordered[ind].next)).previous)==1:
                            self.nodes_ordered[ind].output_var.grad = next(
                                iter(self.nodes_ordered[ind].next)).input_var.grad#.clone()
                        else:
                            self.nodes_ordered[ind].output_var.grad = next(
                                iter(self.nodes_ordered[ind].next)).input_var.grad.clone()
                #pass
                #assert self.nodes_ordered[ind].output_var is next(iter(self.nodes_ordered[ind].next)).input_var#TODO delete this line
                ##self.nodes_ordered[ind].output_var.grad=next(iter(self.nodes_ordered[ind].next)).input_var.grad

            elif len(self.nodes_ordered[ind].next)>1:
                # if not hasattr(self.nodes_ordered[ind].output_var,'grad') or self.nodes_ordered[ind].output_var.grad is None:
                #     self.nodes_ordered[ind].output_var.grad=torch.zeros_like(self.nodes_ordered[ind].output_var)
                # else:
                #     self.nodes_ordered[ind].output_var.grad.zero_()
                # for nxt_node in self.nodes_ordered[ind].next:
                #     self.nodes_ordered[ind].output_var.grad+=nxt_node.input_var.grad

                # nxt_node_list=list(self.nodes_ordered[ind].next)
                # self.nodes_ordered[ind].output_var.grad = nxt_node_list[0].input_var.grad.clone().detach()
                # for i in range(1,len(nxt_node_list)):
                #     self.nodes_ordered[ind].output_var.grad += nxt_node_list[i].input_var.grad.detach()
                with torch.no_grad():
                    next_nodes_iter=iter(self.nodes_ordered[ind].next)
                    self.nodes_ordered[ind].output_var.grad = next(next_nodes_iter).input_var.grad.clone()
                    for next_node in next_nodes_iter:
                        self.nodes_ordered[ind].output_var.grad += next_node.input_var.grad

            if hasattr(self.nodes_ordered[ind].module, 'weight'):
                # self.nodes_ordered[ind].output_var = self.nodes_ordered[ind].module(
                #     self.nodes_ordered[ind].input_var)

                # if ind>0:
                #     self.nodes_ordered[ind].output_var = self.nodes_ordered[ind].module(self.nodes_ordered[ind].input_var)
                # else:
                #     self.intermediate_outputs[ind]=self.layers[ind](self.input_cached)
                # if self.nodes_ordered[ind].module.weight is None or self.nodes_ordered[ind].module.bias is None:
                #     print("Error!")

                bias_present=not self.nodes_ordered[ind].module.bias is None

                grad_outputs = torch.autograd.grad(outputs=self.nodes_ordered[ind].output_var,
                                                   inputs=(self.nodes_ordered[ind].module.weight,self.nodes_ordered[ind].module.bias) if bias_present else self.nodes_ordered[ind].module.weight,
                                                   # inputs=self.intermediate_outputs[0],
                                                   grad_outputs=self.nodes_ordered[ind].output_var.grad
                                                   , retain_graph=True)
                if not average_gradient_of_linear_layers_enhancement:
                    if accumulate_gradients and self.nodes_ordered[ind].module.weight.grad is not None:
                        self.nodes_ordered[ind].module.weight.grad += grad_outputs[0]
                        if bias_present:
                            self.nodes_ordered[ind].module.bias.grad+=grad_outputs[1]
                    else:
                        self.nodes_ordered[ind].module.weight.grad = grad_outputs[0]
                        if bias_present:
                            self.nodes_ordered[ind].module.bias.grad = grad_outputs[1]
                else:
                    grad_outputs2 = torch.autograd.grad(outputs=dependency_graph2.nodes_ordered[ind].output_var,
                                                       inputs=(dependency_graph2.nodes_ordered[ind].module.weight, dependency_graph2.nodes_ordered[ind].module.bias) if bias_present else dependency_graph2.nodes_ordered[ind].module.weight,
                                                       # inputs=self.intermediate_outputs[0],
                                                       grad_outputs=self.nodes_ordered[ind].output_var.grad
                                                       , retain_graph=True)
                    if accumulate_gradients and self.nodes_ordered[ind].module.weight.grad is not None:
                        self.nodes_ordered[ind].module.weight.grad += 0.5*(grad_outputs[0]+grad_outputs2[0])
                        if bias_present:
                            self.nodes_ordered[ind].module.bias.grad+=0.5*(grad_outputs[1]+grad_outputs2[1])
                    else:
                        self.nodes_ordered[ind].module.weight.grad = 0.5 * (grad_outputs[0] + grad_outputs2[0])
                        if bias_present:
                            self.nodes_ordered[ind].module.bias.grad = 0.5 * (grad_outputs[1] + grad_outputs2[1])


            if ind > 0:
                grad_layers = torch.autograd.grad(outputs=self.nodes_ordered[ind].output_var,
                                                  inputs=self.nodes_ordered[ind].input_var,
                                                  # inputs=self.intermediate_outputs[0],
                                                  grad_outputs=self.nodes_ordered[ind].output_var.grad
                                                  , retain_graph=True,
                                                  # allow_unused=True
                                                  )
                self.nodes_ordered[ind].input_var.grad = grad_layers[0]

                if average_gradient_of_linear_layers_enhancement and not (average_gradient_of_nonlinear_layers_enhancement and isinstance(self.nodes_ordered[ind].module, self.nonlinear_simple)):
                    grad_layers2 = torch.autograd.grad(outputs=dependency_graph2.nodes_ordered[ind].output_var,
                                                      inputs=dependency_graph2.nodes_ordered[ind].input_var,
                                                      # inputs=self.intermediate_outputs[0],
                                                      grad_outputs=self.nodes_ordered[ind].output_var.grad
                                                      , retain_graph=True)
                    self.nodes_ordered[ind].input_var.grad = 0.5*(self.nodes_ordered[ind].input_var.grad+grad_layers2[0])

                if average_gradient_of_nonlinear_layers_enhancement:

                    # if isinstance(self.nodes_ordered[ind].module, self.nonlinear_complex):#deprecated
                    #
                    #     log=False
                    #     if log:
                    #         print("----")
                    #         print(self.intermediate_outputs_grad[ind - 1])
                    #     self.gradient_calc(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
                    #                        self.intermediate_outputs_grad[ind - 1],
                    #                        self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
                    #                        self.layers[ind])
                    #
                    #     if log:
                    #         print(self.intermediate_outputs_grad[ind - 1])
                    #     #self.intermediate_outputs_grad[ind - 1]=self.intermediate_outputs_grad[ind - 1]*torch.norm(model2.intermediate_outputs_grad[ind - 1],1)/torch.norm(self.intermediate_outputs_grad[ind - 1],1)
                    if isinstance(self.nodes_ordered[ind].module, self.nonlinear_simple):
                        self.gradient_calc_simple(self.nodes_ordered[ind].input_var, dependency_graph2.nodes_ordered[ind].input_var,
                                           self.nodes_ordered[ind].input_var.grad,
                                           self.nodes_ordered[ind].output_var, self.nodes_ordered[ind].output_var.grad,
                                           self.nodes_ordered[ind].module)


            if (weight_change or gradient_only_modification) and hasattr(self.nodes_ordered[ind].module, 'weight'):
                self.nodes_ordered[ind].module.weight.requires_grad=False
                bias_present=not self.nodes_ordered[ind].module.bias is None
                if bias_present:
                    self.nodes_ordered[ind].module.bias.requires_grad=False

                if gradient_only_modification:
                    self.nodes_ordered[ind].module.weight.grad[:] = torch.abs(
                        dependency_graph2.nodes_ordered[ind].module.weight.grad.detach()) * torch.sign(
                        self.nodes_ordered[ind].module.weight.grad.detach())
                    if bias_present:
                        self.nodes_ordered[ind].module.bias.grad[:] = torch.abs(
                        dependency_graph2.nodes_ordered[ind].module.bias.grad.detach()) * torch.sign(
                        self.nodes_ordered[ind].module.bias.grad.detach())
                else:
                    if step_is_fraction_of_optimizer_denominator!=0.:
                        diff_w=self.nodes_ordered[ind].module.weight.grad.detach()-dependency_graph2.nodes_ordered[ind].module.weight.grad.detach()
                        self.nodes_ordered[ind].module.weight.denominator=diff_w*diff_w+denominator_mul*self.nodes_ordered[ind].module.weight.denominator.detach()

                        self.nodes_ordered[ind].module.weight[:] = dependency_graph2.nodes_ordered[ind].module.weight.detach()+torch.where(self.nodes_ordered[ind].module.weight.denominator!=0.,torch.where(
                            dependency_graph2.nodes_ordered[ind].module.weight.grad!=0.,
                            torch.clip(
                                (self.nodes_ordered[ind].module.weight.grad.detach()-dependency_graph2.nodes_ordered[ind].module.weight.grad.detach())/self.nodes_ordered[ind].module.weight.denominator.detach(),
                                #1./(1.+step_is_fraction_of_optimizer_denominator)-1.,
                                -step_is_fraction_of_optimizer_denominator,
                                step_is_fraction_of_optimizer_denominator)\
                                *(dependency_graph2.nodes_ordered[ind].module.weight.detach()-self.nodes_ordered[ind].module.weight.detach())*torch.abs(self.nodes_ordered[ind].module.weight.denominator.detach()/dependency_graph2.nodes_ordered[ind].module.weight.grad.detach()),
                            0.),0.)
                        if bias_present:
                            diff_b = self.nodes_ordered[ind].module.bias.grad.detach() - \
                                     dependency_graph2.nodes_ordered[ind].module.bias.grad.detach()
                            self.nodes_ordered[ind].module.bias.denominator = diff_b * diff_b + denominator_mul * \
                                                                              self.nodes_ordered[
                                                                                  ind].module.bias.denominator.detach()

                            self.nodes_ordered[ind].module.bias[:] = dependency_graph2.nodes_ordered[ind].module.bias.detach() + torch.where(self.nodes_ordered[ind].module.bias.denominator!=0.,torch.where(
                            dependency_graph2.nodes_ordered[ind].module.bias.grad != 0.,
                            torch.clip(
                                (self.nodes_ordered[ind].module.bias.grad.detach() - dependency_graph2.nodes_ordered[ind].module.bias.grad.detach()) /
                                self.nodes_ordered[ind].module.bias.denominator.detach(),
                                #1. / (1. + step_is_fraction_of_optimizer_denominator) - 1.,
                                 -step_is_fraction_of_optimizer_denominator,
                                step_is_fraction_of_optimizer_denominator) \
                            * (dependency_graph2.nodes_ordered[ind].module.bias.detach() - self.nodes_ordered[ind].module.bias.detach()) * torch.abs(
                                self.nodes_ordered[ind].module.bias.denominator.detach() / dependency_graph2.nodes_ordered[ind].module.bias.grad.detach()),
                            0.),0.)

                    else:
                        self.nodes_ordered[ind].module.weight[:] -= torch.abs(
                            self.nodes_ordered[ind].module.weight.detach() - dependency_graph2.nodes_ordered[ind].module.weight.detach()) * torch.sign(
                            self.nodes_ordered[ind].module.weight.grad.detach())
                        if bias_present:
                            self.nodes_ordered[ind].module.bias[:] -= torch.abs(
                            self.nodes_ordered[ind].module.bias.detach() - dependency_graph2.nodes_ordered[ind].module.bias.detach()) * torch.sign(
                            self.nodes_ordered[ind].module.bias.grad.detach())

                self.nodes_ordered[ind].module.weight.requires_grad = True
                if bias_present:
                    self.nodes_ordered[ind].module.bias.requires_grad = True

    small_num = (torch.finfo(torch.float32).eps)
    @torch.no_grad()
    def gradient_calc_simple(self, layer_input, layer_input_candidate, layer_input_candidate_grad, layer_out,
                             layer_out_candidate_gradient, layer):
        out = layer(layer_input_candidate)
        layer_input_candidate_grad[:] = torch.where(torch.abs(layer_input_candidate - layer_input) > self.small_num,
                                                    (out - layer_out) * layer_out_candidate_gradient /
                                                    (layer_input_candidate - layer_input),
                                                    layer_input_candidate_grad)
    def show_graph(self,indentation='\t'):
        #how_nested=0
        for i in range(len(self.nodes_ordered)):
            layer=self.nodes_ordered[i]
            how_nested=1
            for j in range(i):
                #how_nested+=len(self.nodes_ordered[j].next)
                for nxt_node in self.nodes_ordered[j].next:
                    if self.nodes_ordered.index(nxt_node)>i:
                        how_nested+=1
            print(indentation*how_nested+str(i)+'. '+layer.name+' | connected back to '
                  +str([self.nodes_ordered.index(x) for x in layer.previous])+' | forward to '
                  +str([self.nodes_ordered.index(x) for x in layer.next]))

    def __deepcopy__(self, memo):
        #new_nodes=self.nodes_ordered
        previous_nodes=[]
        for node in self.nodes_ordered:
            previous_nodes.append([self.nodes_ordered.index(x) for x in node.previous])
        next_nodes = []
        for node in self.nodes_ordered:
            next_nodes.append([self.nodes_ordered.index(x) for x in node.next])
        for node in self.nodes_ordered:
            node.previous=None
            node.next=None

        new_nodes=[]
        for node in self.nodes_ordered:
            new_nodes.append(copy.deepcopy(node))

        ind=0
        for node in new_nodes:
            node.previous=set([new_nodes[i] for i in previous_nodes[ind]])
            node.next = set([new_nodes[i] for i in next_nodes[ind]])
            ind+=1
        ind = 0
        for node in self.nodes_ordered:
            node.previous = set([self.nodes_ordered[i] for i in previous_nodes[ind]])
            node.next = set([self.nodes_ordered[i] for i in next_nodes[ind]])
            ind += 1
        c=DependencyGraph(new_nodes[0]) if len(new_nodes)>0 else DependencyGraph([])
        c.order_nodes()
        assert [x.name for x in c.nodes_ordered]==[x.name for x in self.nodes_ordered]
        return c

    def set_require_grad(self,require_grad=True):
        self.require_grad=require_grad

    def input_output_cleanup(self):
        for node in self.nodes_ordered:
            if node is self.first_layer:
                del node.output_var
            elif node is self.last_layer:
                del node.input_var
            else:
                del node.input_var,node.output_var

class NN_Residual(NN):
    # first_layer=None
    # last_layer=None
    dependencyGraph=None

    def __deepcopy__(self, memo):
        c=NN_Residual(copy.deepcopy(self.dependencyGraph))
        return c
        # nodes=c.dependencyGraph.nodes_ordered
        # for i in range(len(self.dependencyGraph.nodes_ordered)):
        #     for j in range(len(c.dependencyGraph.nodes_ordered)):
        #         if self.dependencyGraph.nodes_ordered[i].name
    def __init__(self,modules):
        if isinstance(modules,list):
            if isinstance(modules[len(modules)-1],nn.Softmax):
                modules.pop(len(modules)-1)
            self.init_from_list(modules)
            #NN.__init__(self, modules)
            self.layers = [node.module for node in self.dependencyGraph.nodes_ordered]
            nn.Module.__init__(self)
            self.layer_list=nn.ModuleList(self.layers)

        elif isinstance(modules,DependencyGraph) or type(modules).__name__==DependencyGraph.__name__:
            self.init_from_graph(modules)
            #module_list=[node.module for node in self.dependencyGraph.nodes_ordered]
            #NN.__init__(self, module_list)
            self.layers=[node.module for node in self.dependencyGraph.nodes_ordered]
            nn.Module.__init__(self)
            self.layer_list=nn.ModuleList(self.layers)

    def init_from_list(self,layers:[nn.Module]):
        # #super(NN,self).__init__(layers)
        # #nodes=[]
        # next_node=None
        # for layer in layers:
        #     next_node=DependencyGraphNode(layer,next_node)
        # self.last_layer=next_node
        # self.first_layer=next_node.first()
        self.dependencyGraph=DependencyGraph(layers)


    def init_from_graph(self,graph:DependencyGraph):
        # self.first_layer=node.first()
        # self.last_layer=node.last()
        #
        # node_set=self.first_layer.find_all_nodes()
        self.dependencyGraph=graph
        #node_set = self.dependencyGraph.node_set

    def forward(self, x):
        return self.dependencyGraph.execute_graph(x)
        # raise NotImplementedError
        # self.input_cached=x
        # #layers_num=len(self.layers)
        # #for i in range(layers_num-1):
        # #    x=self.layers[i](x)
        # #output=self.layers[layers_num-1](x)
        # #return output
        #
        # i=0
        # for layer in self.layers:
        #     if self.intermediate_outputs[i] is None:
        #         x=layer(x)
        #         #self.intermediate_outputs[i]=(Variable(x, requires_grad=True))
        #         self.intermediate_outputs[i]=x
        #         self.intermediate_outputs[i].require_grad=True
        #
        #         #self.intermediate_outputs_grad[i]=torch.empty_like(x)
        #     else:
        #         self.intermediate_outputs[i] = layer(x)
        #         x=self.intermediate_outputs[i]
        #     i+=1
        #     if i==len(self.layers)-1 and isinstance(self.layers[len(self.layers)-1],torch.nn.Softmax) and self.training:
        #         break
        # return x

    def backward_grad_correction_with_weight_change2(self, loss, model2, weight_change=True, accumulate_gradients=False,
                                                     gradient_only_modification=False):
        self.dependencyGraph.backward_avg_grad(loss,model2.dependencyGraph,weight_change=weight_change,accumulate_gradients=accumulate_gradients,
                                               gradient_only_modification=gradient_only_modification)
    # def backward_grad_correction_with_weight_change2(self,loss,model2,weight_change=True,accumulate_gradients=False,gradient_only_modification=False):
    #     #start_ind = len(self.intermediate_outputs) - 2 if loss_fn==F.cross_entropy else len(self.intermediate_outputs) - 1
    #     start_ind = len(self.intermediate_outputs) - 1
    #     ind = start_ind#len(self.intermediate_outputs) - 1
    #     grad = torch.autograd.grad(outputs=loss,
    #                                inputs=self.intermediate_outputs[start_ind],
    #                                # inputs=self.intermediate_outputs[0],
    #                                grad_outputs=None
    #                                , retain_graph=True)
    #     self.intermediate_outputs_grad[ind] = grad[0]
    #
    #     if average_gradient_of_loss:
    #         self.intermediate_outputs_grad[ind]=0.5*(grad[0]+model2.intermediate_outputs_grad[ind])
    #
    #
    #     for ind in range(start_ind, -1, -1):
    #         if hasattr(self.layers[ind], 'weight'):
    #             if ind>0:
    #                 self.intermediate_outputs[ind] = self.layers[ind](self.intermediate_outputs[ind-1])
    #             else:
    #                 self.intermediate_outputs[ind]=self.layers[ind](self.input_cached)
    #
    #             grad_outputs = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
    #                                                inputs=[self.layers[ind].weight,self.layers[ind].bias],
    #                                                # inputs=self.intermediate_outputs[0],
    #                                                grad_outputs=self.intermediate_outputs_grad[ind]
    #                                                , retain_graph=True)
    #             if not average_gradient_of_linear_layers_enhancement:
    #                 if accumulate_gradients and self.layers[ind].weight.grad is not None:
    #                     self.layers[ind].weight.grad += grad_outputs[0]
    #                     self.layers[ind].bias.grad+=grad_outputs[1]
    #                 else:
    #                     self.layers[ind].weight.grad = grad_outputs[0]
    #                     self.layers[ind].bias.grad = grad_outputs[1]
    #             else:
    #                 grad_outputs2 = torch.autograd.grad(outputs=model2.intermediate_outputs[ind],
    #                                                    inputs=(model2.layers[ind].weight, model2.layers[ind].bias),
    #                                                    # inputs=self.intermediate_outputs[0],
    #                                                    grad_outputs=self.intermediate_outputs_grad[ind]
    #                                                    , retain_graph=True)
    #                 if accumulate_gradients and self.layers[ind].weight.grad is not None:
    #                     self.layers[ind].weight.grad += 0.5*(grad_outputs[0]+grad_outputs2[0])
    #                     self.layers[ind].bias.grad+=0.5*(grad_outputs[1]+grad_outputs2[1])
    #                 else:
    #                     self.layers[ind].weight.grad = 0.5 * (grad_outputs[0] + grad_outputs2[0])
    #                     self.layers[ind].bias.grad = 0.5 * (grad_outputs[1] + grad_outputs2[1])
    #
    #
    #         if ind > 0:
    #             grad_layers = torch.autograd.grad(outputs=self.intermediate_outputs[ind],
    #                                               inputs=self.intermediate_outputs[ind - 1],
    #                                               # inputs=self.intermediate_outputs[0],
    #                                               grad_outputs=self.intermediate_outputs_grad[ind]
    #                                               , retain_graph=True,
    #                                               # allow_unused=True
    #                                               )
    #             self.intermediate_outputs_grad[ind - 1] = grad_layers[0]
    #
    #             if average_gradient_of_linear_layers_enhancement:
    #                 grad_layers2 = torch.autograd.grad(outputs=model2.intermediate_outputs[ind],
    #                                                   inputs=model2.intermediate_outputs[ind - 1],
    #                                                   # inputs=self.intermediate_outputs[0],
    #                                                   grad_outputs=self.intermediate_outputs_grad[ind]
    #                                                   , retain_graph=True)
    #                 self.intermediate_outputs_grad[ind - 1] = 0.5*(self.intermediate_outputs_grad[ind - 1]+grad_layers2[0])
    #
    #             if average_gradient_of_nonlinear_layers_enhancement:
    #
    #                 if isinstance(self.layers[ind], self.nonlinear_complex):#deprecated
    #
    #                     log=False
    #                     if log:
    #                         print("----")
    #                         print(self.intermediate_outputs_grad[ind - 1])
    #                     self.gradient_calc(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
    #                                        self.intermediate_outputs_grad[ind - 1],
    #                                        self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
    #                                        self.layers[ind])
    #
    #                     if log:
    #                         print(self.intermediate_outputs_grad[ind - 1])
    #                     #self.intermediate_outputs_grad[ind - 1]=self.intermediate_outputs_grad[ind - 1]*torch.norm(model2.intermediate_outputs_grad[ind - 1],1)/torch.norm(self.intermediate_outputs_grad[ind - 1],1)
    #
    #                 elif isinstance(self.layers[ind], self.nonlinear_simple):
    #                     self.gradient_calc_simple(self.intermediate_outputs[ind - 1], model2.intermediate_outputs[ind - 1],
    #                                        self.intermediate_outputs_grad[ind - 1],
    #                                        self.intermediate_outputs[ind], self.intermediate_outputs_grad[ind],
    #                                        self.layers[ind])
    #
    #
    #         if (weight_change or gradient_only_modification) and hasattr(self.layers[ind], 'weight'):
    #             self.layers[ind].weight.requires_grad=False
    #             self.layers[ind].bias.requires_grad=False
    #
    #             if gradient_only_modification:
    #                 self.layers[ind].weight.grad[:] = torch.abs(
    #                     model2.layers[ind].weight.grad.detach()) * torch.sign(
    #                     self.layers[ind].weight.grad.detach())
    #                 self.layers[ind].bias.grad[:] = torch.abs(
    #                     model2.layers[ind].bias.grad.detach()) * torch.sign(
    #                     self.layers[ind].bias.grad.detach())
    #             else:
    #                 if step_is_fraction_of_optimizer_denominator!=0.:
    #                     diff_w=self.layers[ind].weight.grad.detach()-model2.layers[ind].weight.grad.detach()
    #                     self.layers[ind].weight.denominator=diff_w*diff_w+denominator_mul*self.layers[ind].weight.denominator.detach()
    #                     diff_b=self.layers[ind].bias.grad.detach()-model2.layers[ind].bias.grad.detach()
    #                     self.layers[ind].bias.denominator = diff_b*diff_b+denominator_mul*self.layers[ind].bias.denominator.detach()
    #
    #                     self.layers[ind].weight[:] = model2.layers[ind].weight.detach()+torch.where(self.layers[ind].weight.denominator!=0.,torch.where(
    #                         model2.layers[ind].weight.grad!=0.,
    #                         torch.clip(
    #                             (self.layers[ind].weight.grad.detach()-model2.layers[ind].weight.grad.detach())/self.layers[ind].weight.denominator.detach(),
    #                             #1./(1.+step_is_fraction_of_optimizer_denominator)-1.,
    #                             -step_is_fraction_of_optimizer_denominator,
    #                             step_is_fraction_of_optimizer_denominator)\
    #                             *(model2.layers[ind].weight.detach()-self.layers[ind].weight.detach())*torch.abs(self.layers[ind].weight.denominator.detach()/model2.layers[ind].weight.grad.detach()),
    #                         0.),0.)
    #                     self.layers[ind].bias[:] = model2.layers[ind].bias.detach() + torch.where(self.layers[ind].bias.denominator!=0.,torch.where(
    #                         model2.layers[ind].bias.grad != 0.,
    #                         torch.clip(
    #                             (self.layers[ind].bias.grad.detach() - model2.layers[ind].bias.grad.detach()) /
    #                             self.layers[ind].bias.denominator.detach(),
    #                             #1. / (1. + step_is_fraction_of_optimizer_denominator) - 1.,
    #                              -step_is_fraction_of_optimizer_denominator,
    #                             step_is_fraction_of_optimizer_denominator) \
    #                         * (model2.layers[ind].bias.detach() - self.layers[ind].bias.detach()) * torch.abs(
    #                             self.layers[ind].bias.denominator.detach() / model2.layers[ind].bias.grad.detach()),
    #                         0.),0.)
    #
    #                 else:
    #                     self.layers[ind].weight[:] -= torch.abs(
    #                         self.layers[ind].weight.detach() - model2.layers[ind].weight.detach()) * torch.sign(
    #                         self.layers[ind].weight.grad.detach())
    #                     self.layers[ind].bias[:] -= torch.abs(
    #                         self.layers[ind].bias.detach() - model2.layers[ind].bias.detach()) * torch.sign(
    #                         self.layers[ind].bias.grad.detach())
    #
    #             self.layers[ind].weight.requires_grad = True
    #             self.layers[ind].bias.requires_grad = True
    def set_require_grad(self,require_grad=True):
        self.dependencyGraph.set_require_grad(require_grad)

    nodes_with_inputs_to_del=set()
    nodes_with_outputs_to_del=set()
    def delete_all_inputs_and_outputs_that_are_not_nonlinear(self):
        assert not self.dependencyGraph.last_layer.output_var is None
        if not bool(self.nodes_with_inputs_to_del) or not bool(self.nodes_with_outputs_to_del):
            inputs_to_persist=set()
            outputs_to_persist=set()
            all=set()
            for node in self.dependencyGraph.nodes_ordered:
                if isinstance(node.module,self.dependencyGraph.nonlinear_simple):
                    inputs_to_persist.add(node.input_var)
                    outputs_to_persist.add(node.output_var)
                all.add(node.input_var)
                all.add(node.output_var)

            #all=self.dependencyGraph.nodes_ordered
            #inputs_to_persist.discard(next(iter(inputs_to_persist)))
            inputs_to_del=all-inputs_to_persist
            outputs_to_del=all-outputs_to_persist

            for node in self.dependencyGraph.nodes_ordered:
                if node.input_var in inputs_to_del:
                    self.nodes_with_inputs_to_del.add(node)
                if node.output_var in outputs_to_del:
                    self.nodes_with_outputs_to_del.add(node)

        for node in self.nodes_with_inputs_to_del:
            if not node.input_var is None:
                del node.input_var
        for node in self.nodes_with_outputs_to_del:
            if not node.output_var is None:
                del node.output_var

    def input_output_cleanup(self):
        self.dependencyGraph.input_output_cleanup()

    def break_dependency_graph(self):
        self.erase_grad()

        # del self.dependencyGraph.first_layer.input_var
        # del self.dependencyGraph.last_layer.output_var

        # self.input_output_cleanup()
        for node in self.dependencyGraph.nodes_ordered:
            del node.previous
            del node.next
            # del node.module
            # del node
            if hasattr(node.module, 'weight'):
                # del node.module.weight
                tensor = node.module.weight
                tensor.detach()
                tensor.grad = None
                tensor.storage().resize_(0)
                del node.module.weight
                del tensor
            if hasattr(node.module, 'bias'):
                # del node.module.bias
                tensor = node.module.bias
                if tensor is not None:
                    tensor.detach()
                    tensor.grad = None
                    tensor.storage().resize_(0)
                    del node.module.bias
                    del tensor
            del node.module
        del self.dependencyGraph

    @torch.no_grad()
    def grad_l2_norm(self):
        sum_of_squares=0.
        for ind in range(0, len(self.layers)):
            if hasattr(self.layers[ind], 'weight') and hasattr(self.layers[ind].weight, 'grad') and self.layers[ind].weight.grad is not None:
                    sum_of_squares+=torch.sum(torch.square(self.layers[ind].weight.grad))
            if hasattr(self.layers[ind], 'bias') and hasattr(self.layers[ind].bias, 'grad') and self.layers[ind].bias.grad is not None:
                    sum_of_squares+=torch.sum(torch.square(self.layers[ind].bias.grad))
        return sum_of_squares**0.5

    @torch.no_grad()
    def grad_l1_norm(self):
        sum = 0.
        for ind in range(0, len(self.layers)):
            if hasattr(self.layers[ind], 'weight') and hasattr(self.layers[ind].weight, 'grad') and self.layers[
                ind].weight.grad is not None:
                sum += torch.sum(torch.abs(self.layers[ind].weight.grad))
            if hasattr(self.layers[ind], 'bias') and hasattr(self.layers[ind].bias, 'grad') and self.layers[
                ind].bias.grad is not None:
                sum += torch.sum(torch.abs(self.layers[ind].bias.grad))
        return sum

    @torch.no_grad()
    def mul_grad(self,num):
        for ind in range(0, len(self.layers)):
            if hasattr(self.layers[ind], 'weight') and hasattr(self.layers[ind].weight, 'grad') and self.layers[
                ind].weight.grad is not None:
                self.layers[ind].weight.grad*=num
            if hasattr(self.layers[ind], 'bias') and hasattr(self.layers[ind].bias, 'grad') and self.layers[
                ind].bias.grad is not None:
                self.layers[ind].bias.grad*=num

    @torch.no_grad()
    def exp_avg_grad(self,property_name='exp_avg_grad', beta=0.995,grad_model=None):
        if grad_model is None:
            grad_model=self
        #beta=torch.float(beta)
        counter_name=property_name+'_counter'
        if hasattr(self,counter_name):
            setattr(self,counter_name,getattr(self,counter_name)+1)
        else:
            setattr(self, counter_name, 1)

        for ind in range(0, len(self.layers)):
            if hasattr(self.layers[ind], 'weight'):
                if hasattr(self.layers[ind].weight, property_name):# and self.layers[ind].weight.grad is not None:
                    setattr(self.layers[ind].weight, property_name,grad_model.layers[ind].weight.grad+beta*getattr(self.layers[ind].weight, property_name))
                else:
                    setattr(self.layers[ind].weight, property_name, grad_model.layers[ind].weight.grad.clone())
            if hasattr(self.layers[ind], 'bias') and self.layers[ind].bias is not None:
                if hasattr(self.layers[ind].bias, property_name):# and self.layers[ind].bias.grad is not None:
                    setattr(self.layers[ind].bias, property_name,grad_model.layers[ind].bias.grad+beta*getattr(self.layers[ind].bias, property_name))
                else:
                    setattr(self.layers[ind].bias, property_name, grad_model.layers[ind].bias.grad.clone())
    @torch.no_grad()
    def unbias_avg_grad(self,expected_grad='exp_avg_grad',expected_avg_grad='exp_avg_avg_grad', beta=0.995,bias_mul=0.5):
        #beta=torch.float(beta)
        expected_grad_counter=getattr(self,expected_grad+'_counter')
        expected_avg_grad_counter=getattr(self,expected_avg_grad+'_counter')
        expected_grad_mul=bias_mul*(1.-beta)/(1.-beta**expected_grad_counter)
        expected_avg_grad_mul = bias_mul*(1. - beta) / (1. - beta ** expected_avg_grad_counter)

        for ind in range(0, len(self.layers)):
            if hasattr(self.layers[ind], 'weight'):# and hasattr(self.layers[ind].weight, property_name):# and self.layers[ind].weight.grad is not None:
                self.layers[ind].weight.grad+=expected_grad_mul*getattr(self.layers[ind].weight, expected_grad)-expected_avg_grad_mul*getattr(self.layers[ind].weight, expected_avg_grad)
            if hasattr(self.layers[ind], 'bias') and self.layers[ind].bias is not None:# and hasattr(self.layers[ind].bias, property_name):# and self.layers[ind].bias.grad is not None:
                self.layers[ind].bias.grad+=expected_grad_mul*getattr(self.layers[ind].bias, expected_grad)-expected_avg_grad_mul*getattr(self.layers[ind].bias, expected_avg_grad)

    @torch.no_grad()
    def weight_change_as_model_but_directed_by_grad(self, model2,update_only_matching_grad_dimensions=True):
        if update_only_matching_grad_dimensions:
            for ind in range(0, len(self.layers)):
                if hasattr(self.layers[ind], 'weight') and hasattr(self.layers[ind].weight, 'grad') and self.layers[
                    ind].weight.grad is not None:
                    #self.layers[ind].weight-=torch.sign(self.layers[ind].weight.grad)*torch.abs(self.layers[ind].weight-model2.layers[ind].weight)
                    #self.layers[ind].weight[:] = torch.where(self.layers[ind].weight.grad*model2.layers[ind].weight.grad>0.,model2.layers[ind].weight,self.layers[ind].weight*0.5+0.5*model2.layers[ind].weight)
                    self.layers[ind].weight[:] = torch.where(self.layers[ind].weight.grad*model2.layers[ind].weight.grad>0.,model2.layers[ind].weight,self.layers[ind].weight)
                if hasattr(self.layers[ind], 'bias') and hasattr(self.layers[ind].bias, 'grad') and self.layers[
                    ind].bias.grad is not None:
                    #self.layers[ind].bias-=torch.sign(self.layers[ind].bias.grad)*torch.abs(self.layers[ind].bias-model2.layers[ind].bias)
                    #self.layers[ind].bias[:] = torch.where(self.layers[ind].bias.grad*model2.layers[ind].bias.grad>0.,model2.layers[ind].bias,self.layers[ind].bias*0.5+0.5*model2.layers[ind].bias)
                    self.layers[ind].bias[:] = torch.where(self.layers[ind].bias.grad*model2.layers[ind].bias.grad>0.,model2.layers[ind].bias,self.layers[ind].bias)
        else:
            for ind in range(0, len(self.layers)):
                if hasattr(self.layers[ind], 'weight') and hasattr(self.layers[ind].weight, 'grad') and self.layers[
                    ind].weight.grad is not None:
                    self.layers[ind].weight-=torch.sign(self.layers[ind].weight.grad)*torch.abs(self.layers[ind].weight-model2.layers[ind].weight)
                    #self.layers[ind].weight[:] = torch.where(self.layers[ind].weight.grad*model2.layers[ind].weight.grad>0.,model2.layers[ind].weight,self.layers[ind].weight*0.5+0.5*model2.layers[ind].weight)
                    #self.layers[ind].weight[:] = torch.where(self.layers[ind].weight.grad*model2.layers[ind].weight.grad>0.,model2.layers[ind].weight,self.layers[ind].weight)
                if hasattr(self.layers[ind], 'bias') and hasattr(self.layers[ind].bias, 'grad') and self.layers[
                    ind].bias.grad is not None:
                    self.layers[ind].bias-=torch.sign(self.layers[ind].bias.grad)*torch.abs(self.layers[ind].bias-model2.layers[ind].bias)
                    #self.layers[ind].bias[:] = torch.where(self.layers[ind].bias.grad*model2.layers[ind].bias.grad>0.,model2.layers[ind].bias,self.layers[ind].bias*0.5+0.5*model2.layers[ind].bias)
                    #self.layers[ind].bias[:] = torch.where(self.layers[ind].bias.grad*model2.layers[ind].bias.grad>0.,model2.layers[ind].bias,self.layers[ind].bias)

    @torch.no_grad()
    def layerwise_grad_normalization(self, model2,L=2):
        norm_sum=0.
        norm_sum_model2=0.
        #if update_only_matching_grad_dimensions:
        for ind in range(0, len(self.layers)):
            if hasattr(self.layers[ind], 'weight') and hasattr(self.layers[ind].weight, 'grad') and self.layers[
                ind].weight.grad is not None:
                #self.layers[ind].weight-=torch.sign(self.layers[ind].weight.grad)*torch.abs(self.layers[ind].weight-model2.layers[ind].weight)
                #self.layers[ind].weight[:] = torch.where(self.layers[ind].weight.grad*model2.layers[ind].weight.grad>0.,model2.layers[ind].weight,self.layers[ind].weight*0.5+0.5*model2.layers[ind].weight)
                norm=torch.norm(self.layers[ind].weight.grad,p=L)
                norm_model2=torch.norm(model2.layers[ind].weight.grad,p=L)
                if norm!=0.:
                    self.layers[ind].weight.grad[:]=(norm_model2/norm)*self.layers[ind].weight.grad
                    norm_sum+=norm
                norm_sum_model2+=norm_model2
            if hasattr(self.layers[ind], 'bias') and hasattr(self.layers[ind].bias, 'grad') and self.layers[
                ind].bias.grad is not None:
                #self.layers[ind].bias-=torch.sign(self.layers[ind].bias.grad)*torch.abs(self.layers[ind].bias-model2.layers[ind].bias)
                #self.layers[ind].bias[:] = torch.where(self.layers[ind].bias.grad*model2.layers[ind].bias.grad>0.,model2.layers[ind].bias,self.layers[ind].bias*0.5+0.5*model2.layers[ind].bias)
                norm = torch.norm(self.layers[ind].bias.grad, p=L)
                norm_model2 = torch.norm(model2.layers[ind].bias.grad, p=L)
                if norm != 0.:
                    self.layers[ind].bias.grad[:]=(norm_model2/norm)*self.layers[ind].bias.grad
                    norm_sum += norm
                norm_sum_model2 += norm_model2
        return norm_sum,norm_sum_model2

    @torch.no_grad()
    def step_length(self, model2, L=2):
        norm_sum = 0.
        # if update_only_matching_grad_dimensions:
        for ind in range(0, len(self.layers)):
            if hasattr(self.layers[ind], 'weight'):
                norm_sum += torch.norm(self.layers[ind].weight-model2.layers[ind].weight, p=L)
            if hasattr(self.layers[ind], 'bias') and self.layers[ind].bias is not None:
                norm_sum += torch.norm(self.layers[ind].bias-model2.layers[ind].bias, p=L)
        return float(norm_sum)

    @torch.no_grad()
    def mul_update(self, model2, mul):
        for ind in range(0, len(self.layers)):
            if hasattr(self.layers[ind], 'weight'):
                self.layers[ind].weight[:]= mul*(self.layers[ind].weight-model2.layers[ind].weight)+model2.layers[ind].weight
            if hasattr(self.layers[ind], 'bias') and self.layers[ind].bias is not None:
                self.layers[ind].bias[:]=mul*(self.layers[ind].bias-model2.layers[ind].bias)+model2.layers[ind].bias

@torch.no_grad()
def copy_optimizer_params(optim_dest,optim,model_dest,model):
    assert (type(optim_dest)==torch.optim.RMSprop and type(optim)==torch.optim.RMSprop) or (type(optim_dest)==torch.optim.Adam and type(optim)==torch.optim.Adam)
    if not bool(optim.state):
        return
    if type(optim_dest)==torch.optim.Adam:
        for ind in range(len(model_dest.layers)):
            if hasattr(model_dest.layers[ind],'weight'):
                optim_dest.state[model_dest.layers[ind].weight]['exp_avg_sq'][:]=optim.state[model.layers[ind].weight]['exp_avg_sq']
                optim_dest.state[model_dest.layers[ind].weight]['exp_avg'][:] = \
                optim.state[model.layers[ind].weight]['exp_avg']
            if hasattr(model_dest.layers[ind],'bias') and model_dest.layers[ind].bias is not None:
                optim_dest.state[model_dest.layers[ind].bias]['exp_avg_sq'][:]=optim.state[model.layers[ind].bias]['exp_avg_sq']
                optim_dest.state[model_dest.layers[ind].bias]['exp_avg'][:] = \
                optim.state[model.layers[ind].bias]['exp_avg']
    elif type(optim_dest)==torch.optim.RMSprop:
        for ind in range(len(model_dest.layers)):
            if hasattr(model_dest.layers[ind],'weight'):
                optim_dest.state[model_dest.layers[ind].weight]['square_avg'][:]=optim.state[model.layers[ind].weight]['square_avg']

            if hasattr(model_dest.layers[ind],'bias') and model_dest.layers[ind].bias is not None:
                optim_dest.state[model_dest.layers[ind].bias]['square_avg'][:]=optim.state[model.layers[ind].bias]['square_avg']


# def init_param_to_optimizer_param_mapping(self,optimizer):
#     if self.param_to_optimizer_param_mapping_gradient_magnitudes:
#         opt_values = list(optimizer.state.values())
#         key=''
#         if type(optimizer) == torch.optim.Adam:
#             key='exp_avg_sq'
#         elif type(optimizer)==torch.optim.RMSprop:
#             key='square_avg'
#         for i in range(len(opt_values)):
#             self.param_to_optimizer_param_mapping_gradient_magnitudes[opt_values[i]]=opt_values[i][key]

class SavedParams:
    counter=0
    max_counter=1001
    global_init=False
    @classmethod
    def increment_counter(cls):
        cls.counter+=1
        cls.counter=cls.counter%cls.max_counter

    name=''
    def __init__(self,state_data):
        self.name='params'+str(self.counter)
        self.increment_counter()
        self.released=False
        if not SavedParams.global_init:
            SavedParams.global_init=True
            if not os.path.isdir('./tmp'):
                os.mkdir('./tmp')
        self.save(state_data)
    def get_location(self):
        return './tmp/'+self.name
    def save(self,state_data):
        torch.save(state_data, self.get_location())
    def load(self):
        return torch.load(self.get_location())
    def release(self):
        if not self.released:
            self.released=True
            os.remove(self.get_location())
    def __del__(self):
        self.release()

class ModelStateBuffer:
    content=[]
    def __init__(self,size_max):
        self.size_max=size_max
        SavedParams.max_counter=size_max*10+1
    def add(self,state):
        saved_params=SavedParams(state)
        self.content.append(saved_params)
        while len(self.content)>self.size_max:
            self.content[0].release()
            del self.content[0]
    def get_item(self):
        return self.content[0].load()