import utils
import tf_utils
from build_data import build_data
import numpy as np
import tensorflow as tf
import tag

#import tensorflow.compat.v1 as tf

import sys
import os.path

'Train the model on the train set and evaluate on the evaluation and test sets until ' \
'(1) maximum epochs li0 or (2) early stopping break'
def checkInputs():
    if (len(sys.argv) <= 3) or os.path.isfile(sys.argv[0])==False :
        raise ValueError(
            'The configuration file and the timestamp should be specified.')
#应指定配置文件和时间戳。

if __name__ == "__main__":

    checkInputs()

    config=build_data(sys.argv[1])
    

    train_data = utils.HeadData(config.train_id_docs, np.arange(len(config.train_id_docs)))#np.arange
    dev_data = utils.HeadData(config.dev_id_docs, np.arange(len(config.dev_id_docs)))

    test_data = utils.HeadData(config.test_id_docs, np.arange(len(config.test_id_docs)))


    tf.reset_default_graph()        #清除默认图形堆栈并重置全局默认图形
    tf.set_random_seed(1)           #设置图级随机seed

    utils.printParameters(config)    #打印配置

    with tf.Session() as sess:
        embedding_matrix = tf.get_variable('embedding_matrix', shape=config.wordvectors.shape, dtype=tf.float32,
                                           trainable=False).assign(config.wordvectors)
        emb_mtx = sess.run(embedding_matrix)

        model = tf_utils.model(config,emb_mtx,sess)

        obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel, actual_op_rel, score_op_rel = model.run()

        train_step = model.get_train_op(obj)    #

        operations=tf_utils.operations(train_step,obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel, actual_op_rel, score_op_rel)#


        sess.run(tf.global_variables_initializer())#初始化变量

        best_score=0
        nepoch_no_imprv = 0  # for early stopping

        ci = 0  # ci是第几篇文章
        with open(sys.argv[3]+"/es_"+sys.argv[2]+".txt", "w+") as myfile:
            for iter in range(100):  # 控制迭代次数，每迭代一次就把所有的验证集的结果都输出
                # ci = 0
                model.train(train_data,operations,iter)
                
                dev_score = model.evaluate(dev_data, operations, 'dev')
                test_score = model.evaluate(test_data, operations, 'test')
                model.evaluate2(test_data, operations, 'test', ci)  # 此处是测试集的结果
                ci = ci + 1
                myfile.write(str(iter))
           
            # if test_score > best_score:
            #     # model.evaluate2(dev_data, operations, 'dev', ci)  # 此处是验证集的结果
            #     model.evaluate2(test_data, operations, 'test', ci)  # 此处是测试集的结果
            #     ci = ci + 1

            # if test_score>=best_score:
            #     nepoch_no_imprv = 0
            #     best_score = test_score
            #     print ("- Best test score {} so far in {} epoch".format(test_score,iter))

            # else:
            #     nepoch_no_imprv += 1
            #     if nepoch_no_imprv >= config.nepoch_no_imprv:

            #         print ("- early stopping {} epochs without " \
            #                          "improvement".format(nepoch_no_imprv))

            #         with open(sys.argv[3]+"/es_"+sys.argv[2]+".txt", "w+") as myfile:
            #             myfile.write(str(iter))
            #             myfile.close()

            #         break
           
            #saver.save(sess, "net/my_net.ckpt")  原有

        





