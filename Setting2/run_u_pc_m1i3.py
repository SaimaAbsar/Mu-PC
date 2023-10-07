#!/usr/bin/env python
import click
import tf_rmtpp
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
import tempfile
import matplotlib.pyplot as plt
import numpy as np

# Mu_PC for i = 1 (C = 1)


def_opts = tf_rmtpp.rmtpp_core_u_pc_m1i3.def_opts

@click.command()
@click.argument('event_train_file_x')
@click.argument('event_test_file_x')
@click.argument('event_train_file_z0')
@click.argument('event_test_file_z0')
@click.argument('event_train_file_z1')
@click.argument('event_test_file_z1')
@click.argument('event_train_file_z2')
@click.argument('event_test_file_z2')
@click.argument('event_train_file_y')
@click.argument('event_test_file_y')
@click.option('--summary', 'summary_dir', help='Which folder to save summaries to.', default=None)
@click.option('--epochs', 'num_epochs', help='How many epochs to train for.', default=1)
@click.option('--restart/--no-restart', 'restart', help='Can restart from a saved model from the summary folder, if available.', default=False)
@click.option('--train-eval/--no-train-eval', 'train_eval', help='Should evaluate the model on training data?', default=False)
@click.option('--test-eval/--no-test-eval', 'test_eval', help='Should evaluate the model on test data?', default=True)
@click.option('--scale', 'scale', help='Constant to scale the time fields by.', default=1.0)
@click.option('--batch-size', 'batch_size', help='Batch size.', default=1)
@click.option('--bptt', 'bptt', help='Series dependence depth.', default=def_opts.bptt)
@click.option('--init-learning-rate', 'learning_rate', help='Initial learning rate.', default=def_opts.learning_rate)
@click.option('--cpu-only/--no-cpu-only', 'cpu_only', help='Use only the CPU.', default=def_opts.cpu_only)
def cmd(event_train_file_x, event_test_file_x, event_train_file_z0,
        event_test_file_z0, event_train_file_z1,
        event_test_file_z1, event_train_file_z2,
        event_test_file_z2, event_train_file_y, event_test_file_y,
        summary_dir, num_epochs, restart, train_eval, test_eval, scale,
        batch_size, bptt, learning_rate, cpu_only):
    """Read data from EVENT_TRAIN_FILE, TIME_TRAIN_FILE and try to predict the values in EVENT_TEST_FILE, TIME_TEST_FILE."""
    data = tf_rmtpp.utils_m1i3.read_data(
        event_train_file_x=event_train_file_x,
        event_test_file_x=event_test_file_x,
        event_train_file_z0=event_train_file_z0,
        event_test_file_z0=event_test_file_z0,
        event_train_file_z1=event_train_file_z1,
        event_test_file_z1=event_test_file_z1,
        event_train_file_z2=event_train_file_z2,
        event_test_file_z2=event_test_file_z2,
        event_train_file_y=event_train_file_y,
        event_test_file_y=event_test_file_y
    )
    '''
    data['train_time_out_seq'] /= scale
    data['train_time_in_seq'] /= scale
    data['test_time_out_seq'] /= scale
    data['test_time_in_seq'] /= scale
    data['tj_out_seq'] /= scale
    data['tj_in_seq'] /= scale
    data['tjTest_out_seq'] /= scale
    data['tjTest_in_seq'] /= scale
    '''
    

    tf.reset_default_graph()
    sess = tf.Session()

    tf_rmtpp.utils_m1i3.data_stats(data)

    rmtpp_mdl = tf_rmtpp.rmtpp_core_u_pc_m1i3.RMTPP(
        sess=sess,
        num_categories=data['num_categories'],
        summary_dir=summary_dir if summary_dir is not None else tempfile.mkdtemp(),
        batch_size=batch_size,
        bptt=bptt,
        learning_rate=learning_rate,
        cpu_only=cpu_only,
        _opts=tf_rmtpp.rmtpp_core_u_pc_m1i3.def_opts
    )

    # TODO: The finalize here has to be false because tf.global_variables()
    # creates a new graph node (why?). Hence, need to be extra careful while
    # saving the model.
    rmtpp_mdl.initialize(finalize=False)
    last_epoch = 0
    llt = []
    llp = []
    t_l = []
    m_l = []
    for epoch in range(last_epoch, last_epoch + num_epochs):
        #self.rs.shuffle(idxes)

        print("Starting epoch...", epoch)
        

        l_t = rmtpp_mdl.train(training_data=data, restart=restart,
                    with_summaries=summary_dir is not None,
                    num_epochs=num_epochs, with_evals=True)
        l_p = rmtpp_mdl.predict_t(num_epochs, data['test_event_in_seq'], data['test_event_out_seq'])
        #for j in l_p[0]:
        llp.append(l_p)
        
    last_epoch += num_epochs
    llp = [x for x in llp if str(x) != 'nan']
    
   ###########------write loss value to file --------###################
    with open('Loss_test_model1_i3.txt', 'w') as output:
        #for i in range(len(batch_loss)):
 
        output.write(str(np.min(llp)))
        #x = batch_loss
        #output.write(str(x))
    ######################################################################   
  

    if train_eval:
        print('\nEvaluation on training data:')
        train_time_preds = rmtpp_mdl.predict_train(num_epochs=num_epochs, data=data)
        rmtpp_mdl.eval(train_event_preds, data['train_event_out_seq'])
        print()

    if test_eval:
        print('\nEvaluation on testing data:')
        test_event_preds = rmtpp_mdl.predict_test(num_epochs=num_epochs, data=data)
        t = rmtpp_mdl.eval(test_event_preds, data['test_event_out_seq'])
        #print(np.shape(t), np.shape(e))
        '''
        #print (chi_sq)
        ###########------write loss value to file --------###################
        with open('Loss_test_model1_i0.txt', 'w') as output:
            #for i in range(len(batch_loss)):
     
            output.write(str(np.sum(chi_sq)))
            #x = batch_loss
            #output.write(str(x))
        ######################################################################   
        
        ###########------write loss value to file --------###################
        with open('Time_test_model1_i2.txt', 'w') as output:
            mstr = ''
            for ix in t[0]:
                mstr += str(ix)+" "     
            output.write(mstr)
            output.write("\n")
                
            #x = batch_loss
            #output.write(str(x))
        ###################################################################### 
        ###########------write loss value to file --------###################
        with open('Event_test_model1_i2.txt', 'w') as output:
            mstr = ''
            for ix in e[0]:
                mstr += str(ix)+" "     
            output.write(mstr)
            output.write("\n")
                
            #x = batch_loss
            #output.write(str(x))
        ###################################################################### 
        '''
    print()

if __name__ == '__main__':
    cmd()
