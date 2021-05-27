import numpy as np
import pandas as pd
import seaborn as sns
import pysynphot as S

#Need 814 ACS/HRC
#Need 555 ACS/HRC
#Need 330 ACS/HRC
#Need 775 WFC3/UVIS

bands_needed = np.array(['acs,hrc,f814w','acs,hrc,f555w','acs,hrc,f330w','wfc3,uvis1,f775w'])
for band in bands_needed:
    bp = S.ObsBandpass(band)
    waves = bp.wave
    through = bp.throughput
    throughputFile = pd.DataFrame({'0':waves, '1':through})
    band_fn = band.split(",")
    throughputFile.to_csv("~/miniconda3/envs/prospector/lib/python3.8/site-packages/sedpy/data/filters/%s_%s_%s.par"%(band_fn[0], band_fn[1],band_fn[2]),header=False,index=False)
