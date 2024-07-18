import json
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import StrMethodFormatter
from tqdm import tqdm

from utils import generate_iv_samples, mutual_information, mutual_information_vec, generate_conditional_dist, entropy, conditional_latent_search, conditional_latent_search_iv, generate_invalid_iv_samples, optimization_entr, get_natural_bounds, entropy_vec, mutual_information_vec, get_tp_bounds


'''
Construct the distributions for the asia dataset
probability ( asia ) {
  table 0.01, 0.99;
}
probability ( tub | asia ) {
  (yes) 0.05, 0.95;
  (no) 0.01, 0.99;
}
probability ( smoke ) {
  table 0.5, 0.5;
}
probability ( lung | smoke ) {
  (yes) 0.1, 0.9;
  (no) 0.01, 0.99;
}
probability ( bronc | smoke ) {
  (yes) 0.6, 0.4;
  (no) 0.3, 0.7;
}
probability ( either | lung, tub ) {
  (yes, yes) 1.0, 0.0;
  (no, yes) 1.0, 0.0;
  (yes, no) 1.0, 0.0;
  (no, no) 0.0, 1.0;
}
probability ( xray | either ) {
  (yes) 0.98, 0.02;
  (no) 0.05, 0.95;
}
probability ( dysp | bronc, either ) {
  (yes, yes) 0.9, 0.1;
  (no, yes) 0.7, 0.3;
  (yes, no) 0.8, 0.2;
  (no, no) 0.1, 0.9;
}


'''

pSmoke = np.array([0.5, 0.5])
pLung_Smoke = np.array([[0.1, 0.9], [0.01, 0.99]]).transpose(1,0)

pLungSmoke = np.einsum('jk, k -> jk', pLung_Smoke, pSmoke)

pBronc_Smoke = np.array([[0.6, 0.4], [0.3, 0.7]]).transpose(1,0)
pEither_LungTub = np.array([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]).transpose(2,1,0)
pDysp_BroncEither = np.array([[[0.9, 0.1], [0.7, 0.3]], [[0.8, 0.2], [0.1, 0.9]]]).transpose(2,1,0)
pTub_Asia = np.array([[0.05, 0.95], [0.01, 0.99]]).transpose(1,0)
pAsia = np.array([0.01, 0.99])

pDyspBroncEitherLungSmokeTubAsia = np.einsum('dbe, bs, elt, ls, s, ta, a -> dbelsta', pDysp_BroncEither, pBronc_Smoke, pEither_LungTub, pLung_Smoke, pSmoke, pTub_Asia, pAsia)

pDyspBroncEither = pDyspBroncEitherLungSmokeTubAsia.sum(axis=(3,4,5,6))

pDyspBronc = pDyspBroncEither.sum(axis=2)
pDyspEither = pDyspBroncEither.sum(axis=1)

pBronc = pDyspBronc.sum(axis=0)
pEither = pDyspEither.sum(axis=0)

pLung = pLungSmoke.sum(axis=1)
HL = entropy(pLung)
HS = entropy(pSmoke)
HB = entropy(pBronc)

pDyspEitherTub = pDyspBroncEitherLungSmokeTubAsia.sum(axis=(1,3,4,6))


pDyspTubEither = pDyspEitherTub.transpose(0,2,1)
pDyspTub_Either = np.einsum('ijk, k ->ijk', pDyspTubEither, np.divide(1, pEither, where=(pEither>0)))


pEitherTub = pDyspEitherTub.sum(axis=0)
pTub = pEitherTub.sum(axis=0)

## Compute the conditional common entropy
pDyspEitherAsia = pDyspBroncEitherLungSmokeTubAsia.sum(axis=(1,3,4,5))
pDyspAsiaEither = pDyspEitherAsia.transpose(0,2,1)
pDyspAsia_Either = np.einsum('ijk, k ->ijk', pDyspAsiaEither, np.divide(1, pEither, where=(pEither>0)))

## Compute the bounds

pDyspBroncEitherLungTubAsia_Smoke  =np.einsum('dbe, bs, elt, ls, ta, a -> dbeltas', pDysp_BroncEither, pBronc_Smoke, pEither_LungTub, pLung_Smoke, pTub_Asia, pAsia)

pDyspBronc_Smoke = pDyspBroncEitherLungTubAsia_Smoke.sum(axis=(2,3,4,5))


pDyspBroncEitherLungSmokeAsiaTub = pDyspBroncEitherLungSmokeTubAsia.transpose(0,1,2,3,4,6,5)
pDyspBroncEitherLungSmokeAsia_Tub = np.einsum('dbelsat, t->dbelsat', pDyspBroncEitherLungSmokeAsiaTub, np.divide(1, pTub, where=(pTub>0)))
pDyspEither_Tub = pDyspBroncEitherLungSmokeAsia_Tub.sum(axis=(1,3,4,5))

for x_idx in [0,1]:
  for y_idx in [0,1]:
    entr_bounds = optimization_entr(pDyspEither_Tub, pTub, y_idx=y_idx, x_idx=x_idx, entr=HL, izy=0)
    natural_bounds = get_natural_bounds(pDyspEither_Tub, y_idx=y_idx, x_idx=x_idx)
    tp_bounds = get_tp_bounds(pDyspEither, y_idx=y_idx, x_idx=x_idx)

    print(f"P(y{x_idx} | x{y_idx}):")
    print(f"Natural bounds: {natural_bounds}")
    print(f"TP bounds: {tp_bounds}")
    print(f"Entr bounds: {entr_bounds}")
    print("")
    


