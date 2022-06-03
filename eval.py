import os
import subprocess
import csv

dir_path = 'checkpoint_listops'
mapping = {}
for model_path in os.listdir(dir_path):
    params = model_path.split('-')
    if len(params) == 13:
        mode, temp, seed, epoch = '-'.join(params[1:3]), params[4], int(params[6]), float(params[8])
    else:
        mode, temp, seed, epoch = params[1], params[3], int(params[5]), float(params[7])
    model_name = '{}{}{}'.format(mode, temp, seed)
    if model_name not in mapping or (model_name in mapping and epoch > mapping[model_name]['epoch']):
        mapping[model_name] = {'mode':mode, 'temp':temp, 'seed':seed, 'epoch':epoch, 'full_path':model_path}

csv_writer = csv.writer(open('test_result.csv', 'w'))

for temp in [1.0, 0.1, 0.01]:
    for mode in ['gumbel', 'rao_gumbel', 'gap-0.8', 'gap-1.0', 'gap-1.2', 'gap-p']:
        row = []
        for seed in [1, 2, 3, 4]:
            for k, v in mapping.items():
                if v['mode'] == mode and v['temp'] == str(temp) and v['seed'] == seed:
                    bashCommand = "python -m nlp.evaluate --word-dim 300 --hidden-dim 300 --clf-hidden-dim 300 --clf-num-layers 1 --device cuda --leaf-rnn --dropout 0.5 --lower --mode {} --task listops --temperature {} --model {}".format(v['mode'], v['temp'], os.path.join(dir_path, v['full_path']))
                    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                    output, error = process.communicate()
                    output = output.decode().split('\n')
                    loss, acc = float(output[0].split(':')[-1].strip()), float(output[-2].split(':')[-1].strip())
                    row += [loss, acc]
        csv_writer.writerow(row)
        print('Finish {} {}'.format(temp, mode))
