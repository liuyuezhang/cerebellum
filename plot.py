import wandb
import re
import pandas as pd

api = wandb.Api()
runs = api.runs("liuyuezhang/bp-rnn")
print("Found %i" % len(runs))


df = pd.DataFrame()
for run in runs:
    m = re.match('(.*)_(.*)_(.*)', run.name)
    env = m.group(1)
    method = m.group(2)
    seed = m.group(3)

    temp = run.history()
    temp['method'] = method

    if method == 'pytorch-rmsprop':
        df = df.append(temp, ignore_index=True)

print(df)

