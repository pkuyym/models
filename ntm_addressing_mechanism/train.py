import paddle.v2 as paddle
from ntm_conf import gru_encoder_decoder
import sys


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    dict_size = 30000

    cost = gru_encoder_decoder(
        src_dict_dim=dict_size,
        trg_dict_dim=dict_size,
        is_generating=False,
        is_hybrid_addressing=True)

    parameters = paddle.parameters.create(cost)

    optimizer = paddle.optimizer.Adam(
        learning_rate=5e-4,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=2500),
        learning_rate_decay_a=0.0,
        learning_rate_decay_b=0.0,
        gradient_clipping_threshold=25)

    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    wmt14_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size, src_init=True),
            buf_size=8192),
        batch_size=5)

    def event_handler(event):
        if isinstance(event, paddle.event.EndPass):
            model_name = './models/model_pass_%05d.tar.gz' % event.pass_id
            print('Save model to %s !' % model_name)
            with gzip.open(model_name, 'w') as f:
                parameters.to_tar(f)

        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 10 == 0:
                print("\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

    # start to train
    trainer.train(
        reader=wmt14_reader, event_handler=event_handler, num_passes=2)


if __name__ == '__main__':
    main()
