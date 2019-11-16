# !svn export https://github.com/sauxpa/Quant/trunk/ito_diffusions
    
import argparse
from ito_diffusions import *

# Get arguments
parser=argparse.ArgumentParser()
parser.add_argument('--T', default=1.0, type=float, help='Maturity')
parser.add_argument('--scheme_steps', default=250, type=int, help='Number of recorded times')
parser.add_argument('--n_sim', default=1000, type=int, help='Number of price paths')
parser.add_argument('--x0', default=100.0, type=float, help='Starting price')
parser.add_argument('--H', default=0.5, type=float, help='Hurst index')
parser.add_argument('--vol', default=1.0, type=float, help='Drift')
parser.add_argument('--drift', default=0.0, type=float, help='Volatility')
parser.add_argument('--fname', default='', type=str, help='File name')

args = parser.parse_args()
T = args.T
scheme_steps = args.scheme_steps
n_sim = args.n_sim
x0 = args.x0
H = args.H
drift = args.drift
vol = args.vol
fname = args.fname

if not fname:
    fname = 'fbm_0{}.csv'.format(int(1000*H))

X = FBM(x0=x0, T=T, scheme_steps=scheme_steps, drift=drift, vol=vol, H=H)

dfs = []
for i in range(n_sim):
    df = X.simulate()
    df.columns = [str(i)]
    dfs.append(df)

df = pd.concat(dfs, axis=1)

df.to_csv(fname, index=False)
