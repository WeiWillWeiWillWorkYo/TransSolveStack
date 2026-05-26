# SuiteSparse Matrix Subset Selection

- status: `ready`
- candidates_considered: `2904`
- candidates_after_filter: `1854`
- selected_matrices: `64`
- total_selected_archive_size_gb: `0.153`
- source_index_path: `/mnt/tss_external/TransSolveStack/datasets/suitesparse_full/index/matrix_manifest.jsonl`

## Size Buckets

| bucket | count |
|---|---:|
| medium | 3 |
| small | 12 |
| tiny | 49 |

## Kinds

| kind | count |
|---|---:|
| 2D/3D problem | 1 |
| 2D/3D problem sequence | 1 |
| acoustics problem | 1 |
| chemical process simulation problem | 1 |
| chemical process simulation problem sequence | 1 |
| circuit simulation problem | 1 |
| circuit simulation problem sequence | 1 |
| combinatorial problem | 1 |
| computational chemistry problem | 1 |
| computational fluid dynamics | 1 |
| computational fluid dynamics problem | 1 |
| computational fluid dynamics problem sequence | 1 |
| computer graphics/vision problem | 1 |
| counter-example problem | 1 |
| directed graph | 1 |
| directed graph with communities | 1 |
| directed multigraph | 1 |
| directed temporal multigraph | 1 |
| directed weighted graph | 1 |
| directed weighted graph sequence | 1 |
| directed weighted random graph | 1 |
| directed weighted temporal graph | 1 |
| directed weighted temporal multigraph | 1 |
| duplicate computational fluid dynamics problem | 1 |
| duplicate economic problem | 1 |
| duplicate electromagnetics problem | 1 |
| duplicate materials problem | 1 |
| duplicate model reduction problem | 1 |
| duplicate optimization problem | 1 |
| duplicate structural problem | 1 |
| duplicate thermal problem | 1 |
| duplicate undirected random graph | 1 |
| economic problem | 1 |
| eigenvalue/model reduction problem | 1 |
| electromagnetics problem | 1 |
| frequency-domain circuit simulation problem | 1 |
| least squares problem | 1 |
| linear programming problem | 1 |
| materials problem | 1 |
| model reduction problem | 1 |
| optimal control problem | 1 |
| optimization problem | 1 |
| optimization problem sequence | 1 |
| power network problem | 1 |
| power network problem sequence | 1 |
| random 2D/3D problem | 1 |
| random undirected graph | 1 |
| random unweighted graph | 1 |
| robotics problem | 1 |
| semiconductor device problem | 1 |
| semiconductor device problem sequence | 1 |
| semiconductor process problem | 1 |
| statistical/mathematical problem | 1 |
| structural problem | 1 |
| structural problem sequence | 1 |
| subsequent 2D/3D problem | 1 |
| subsequent circuit simulation problem | 1 |
| subsequent computational fluid dynamics problem | 1 |
| subsequent optimization problem | 1 |
| subsequent power network problem | 1 |
| subsequent semiconductor device problem | 1 |
| subsequent theoretical/quantum chemistry problem | 1 |
| theoretical/quantum chemistry problem | 1 |
| theoretical/quantum chemistry problem sequence | 1 |

## Selected Matrices

| rank | matrix | shape | nnz | bucket | kind | archive_mb |
|---:|---|---:|---:|---|---|---:|
| 1 | suitesparse:HB/curtis54 | 54x54 | 291 | tiny | 2D/3D problem | 0.001 |
| 2 | suitesparse:HB/fs_183_1 | 183x183 | 998 | tiny | 2D/3D problem sequence | 0.009 |
| 3 | suitesparse:HB/young3c | 841x841 | 3988 | tiny | acoustics problem | 0.014 |
| 4 | suitesparse:Grund/b1_ss | 7x7 | 15 | tiny | chemical process simulation problem | 0.001 |
| 5 | suitesparse:Zitney/extr1b | 2836x2836 | 10965 | tiny | chemical process simulation problem sequence | 3.569 |
| 6 | suitesparse:Hamrle/Hamrle1 | 32x32 | 98 | tiny | circuit simulation problem | 0.001 |
| 7 | suitesparse:Sandia/oscil_dcop_01 | 430x430 | 1544 | tiny | circuit simulation problem sequence | 0.009 |
| 8 | suitesparse:JGD_Trefethen/Trefethen_20b | 19x19 | 147 | tiny | combinatorial problem | 0.001 |
| 9 | suitesparse:Negre/dendrimer | 730x730 | 63024 | tiny | computational chemistry problem | 0.212 |
| 10 | suitesparse:Goodwin/Goodwin_010 | 1182x1182 | 32282 | tiny | computational fluid dynamics | 0.165 |
| 11 | suitesparse:FIDAP/ex5 | 27x27 | 279 | tiny | computational fluid dynamics problem | 0.001 |
| 12 | suitesparse:Bai/cdde1 | 961x961 | 4681 | tiny | computational fluid dynamics problem sequence | 0.019 |
| 13 | suitesparse:MathWorks/tomography | 500x500 | 28726 | tiny | computer graphics/vision problem | 0.277 |
| 14 | suitesparse:HB/jgl009 | 9x9 | 50 | tiny | counter-example problem | 0.001 |
| 15 | suitesparse:HB/ibm32 | 32x32 | 126 | tiny | directed graph | 0.001 |
| 16 | suitesparse:SNAP/email-Eu-core | 1005x1005 | 25571 | tiny | directed graph with communities | 0.068 |
| 17 | suitesparse:Pajek/GD01_c | 33x33 | 135 | tiny | directed multigraph | 0.001 |
| 18 | suitesparse:SNAP/CollegeMsg | 1899x1899 | 20296 | tiny | directed temporal multigraph | 0.417 |
| 19 | suitesparse:vanHeukelum/cage3 | 5x5 | 19 | tiny | directed weighted graph | 0.001 |
| 20 | suitesparse:SNAP/as-caida | 31379x31379 | 106762 | small | directed weighted graph sequence | 39.906 |
| 21 | suitesparse:Simon/appu | 14000x14000 | 1853104 | medium | directed weighted random graph | 21.363 |
| 22 | suitesparse:SNAP/soc-sign-bitcoin-alpha | 3783x3783 | 24186 | tiny | directed weighted temporal graph | 0.339 |
| 23 | suitesparse:SNAP/wiki-RfA | 11380x11380 | 188077 | small | directed weighted temporal multigraph | 14.733 |
| 24 | suitesparse:GHS_psdef/copter2 | 55476x55476 | 759952 | small | duplicate computational fluid dynamics problem | 1.017 |
| 25 | suitesparse:Mulvey/pfinan512 | 74752x74752 | 596992 | small | duplicate economic problem | 0.895 |
| 26 | suitesparse:Li/pli | 22695x22695 | 1350309 | medium | duplicate electromagnetics problem | 1.499 |
| 27 | suitesparse:Boeing/pcrystk02 | 13965x13965 | 968583 | small | duplicate materials problem | 0.934 |
| 28 | suitesparse:Oberwolfach/t2dal_e | 4257x4257 | 4257 | tiny | duplicate model reduction problem | 0.040 |
| 29 | suitesparse:GHS_psdef/torsion1 | 40000x40000 | 197608 | small | duplicate optimization problem | 0.380 |
| 30 | suitesparse:HB/bcsstk07 | 420x420 | 7860 | tiny | duplicate structural problem | 0.027 |
| 31 | suitesparse:HB/lshp1009 | 1009x1009 | 6865 | tiny | duplicate thermal problem | 0.009 |
| 32 | suitesparse:Gset/G17 | 800x800 | 9334 | tiny | duplicate undirected random graph | 0.013 |
| 33 | suitesparse:Grund/poli | 4008x4008 | 8188 | tiny | economic problem | 0.034 |
| 34 | suitesparse:Rommes/S10PI_n1 | 528x528 | 1317 | tiny | eigenvalue/model reduction problem | 0.011 |
| 35 | suitesparse:Bai/bfwb62 | 62x62 | 342 | tiny | electromagnetics problem | 0.001 |
| 36 | suitesparse:ATandT/onetone2 | 36057x36057 | 222596 | small | frequency-domain circuit simulation problem | 1.304 |
| 37 | suitesparse:HB/ash85 | 85x85 | 523 | tiny | least squares problem | 0.001 |
| 38 | suitesparse:Meszaros/iprob | 3001x3001 | 9000 | tiny | linear programming problem | 0.042 |
| 39 | suitesparse:HB/arc130 | 130x130 | 1037 | tiny | materials problem | 0.013 |
| 40 | suitesparse:Oberwolfach/LFAT5 | 14x14 | 46 | tiny | model reduction problem | 0.001 |
| 41 | suitesparse:VDOL/spaceStation_1 | 99x99 | 927 | tiny | optimal control problem | 0.009 |
| 42 | suitesparse:HB/zenios | 2873x2873 | 1314 | tiny | optimization problem | 0.049 |
| 43 | suitesparse:HB/shl_0 | 663x663 | 1687 | tiny | optimization problem sequence | 0.006 |
| 44 | suitesparse:HB/bcspwr01 | 39x39 | 131 | tiny | power network problem | 0.001 |
| 45 | suitesparse:HB/gemat11 | 4929x4929 | 33108 | tiny | power network problem sequence | 0.284 |
| 46 | suitesparse:GHS_psdef/wathen100 | 30401x30401 | 471601 | small | random 2D/3D problem | 2.317 |
| 47 | suitesparse:DIMACS10/smallworld | 100000x100000 | 999996 | small | random undirected graph | 1.227 |
| 48 | suitesparse:DIMACS10/vsp_c-30_data_data | 11023x11023 | 124368 | small | random unweighted graph | 0.147 |
| 49 | suitesparse:Morandini/robot | 120x120 | 870 | tiny | robotics problem | 0.005 |
| 50 | suitesparse:HB/will57 | 57x57 | 281 | tiny | semiconductor device problem | 0.001 |
| 51 | suitesparse:Wang/wang1 | 2903x2903 | 19093 | tiny | semiconductor device problem sequence | 0.097 |
| 52 | suitesparse:VLSI/ss1 | 205282x205282 | 845089 | medium | semiconductor process problem | 5.139 |
| 53 | suitesparse:HB/gent113 | 113x113 | 655 | tiny | statistical/mathematical problem | 0.002 |
| 54 | suitesparse:HB/lap_25 | 25x25 | 169 | tiny | structural problem | 0.001 |
| 55 | suitesparse:TKK/t2d_q4 | 9801x9801 | 87025 | tiny | structural problem sequence | 49.631 |
| 56 | suitesparse:HB/fs_183_6 | 183x183 | 1000 | tiny | subsequent 2D/3D problem | 0.009 |
| 57 | suitesparse:Sandia/oscil_dcop_02 | 430x430 | 1544 | tiny | subsequent circuit simulation problem | 0.013 |
| 58 | suitesparse:Bai/cdde4 | 961x961 | 4681 | tiny | subsequent computational fluid dynamics problem | 0.017 |
| 59 | suitesparse:HB/shl_200 | 663x663 | 1726 | tiny | subsequent optimization problem | 0.007 |
| 60 | suitesparse:HB/gemat12 | 4929x4929 | 33044 | tiny | subsequent power network problem | 0.283 |
| 61 | suitesparse:Wang/wang2 | 2903x2903 | 19093 | tiny | subsequent semiconductor device problem | 0.098 |
| 62 | suitesparse:Nemeth/nemeth02 | 9506x9506 | 394808 | small | subsequent theoretical/quantum chemistry problem | 2.029 |
| 63 | suitesparse:PARSEC/Si2 | 769x769 | 17801 | tiny | theoretical/quantum chemistry problem | 0.055 |
| 64 | suitesparse:Nemeth/nemeth01 | 9506x9506 | 725054 | small | theoretical/quantum chemistry problem sequence | 4.049 |
