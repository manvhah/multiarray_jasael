import numpy as np
from torch.utils.data import Dataset
import pickle
import torch

def get_measurement_array_from_dict(dd, extract_all = False, Narrays=7):
    sources = [key for key in dd.keys()]
    Nsources = len(sources)

    if extract_all: # extract all sources
        Nsamples, NXYZchannels = dd[sources[0]].shape
        ddarray = np.empty((Nsamples,Nsources*NXYZchannels))
        for ss,source in enumerate(sources):
            for ii,chan in enumerate(dd[source].keys()[1:]):
                ddarray[:,ss*NXYZchannels+ii] = dd[source][chan]
    else:
        # for each channel, pick the first not-completely-nan source
        def gen_chankey(ii,ss):
            return ['X_{:d}_{:d}'.format(ss,ii),
                    'Y_{:d}_{:d}'.format(ss,ii),
                    'Z_{:d}_{:d}'.format(ss,ii)]
        Nsamples = dd[sources[0]].shape[0]
        ddarray = np.empty((Nsamples,Narrays*3))
        for ii in range(Narrays): # for each array, array 0 is not valid
            chankey = gen_chankey(ii+1,0)
            init = True
            valid = not np.all(np.isnan(dd[sources[0]][chankey]))
            for ss in range(4): # traverse along sources
                chankey = gen_chankey(ii+1,ss)
                if ss == 0:
                    xyz = dd[sources[ss]][chankey].copy()
                else:
                    xyz[vcheck] = dd[sources[ss]][chankey][vcheck]
                vcheck = np.isnan(xyz)
                valid = not np.any(vcheck)
            ddarray[:,ii*3:ii*3+3] = xyz
            
    # set nans to zero
    ddarray = np.nan_to_num(ddarray) 
    # remove all points picked up by less than two arrays
    ddarray = ddarray[np.sum(np.abs(ddarray) > 0,axis=1) > 3] 
    # filter repetitions
    grad = np.hstack([np.ones(1), np.linalg.norm(np.diff(ddarray,axis=0), axis=1)])
    ddarray = ddarray[grad>0] 

    return ddarray.astype(np.float32)

def delft(x,f,t): 
    if f is None:
        return x[t:]
    elif t is None:
        return x[:f]
    else:
        return np.vstack([x[:f],x[t:]])
    
ZREF = dict(
    # offset for speaker size, acoustic center approx + 5cm above location
    table    = 0.79,
    counter  = 0.92,
    ground   = 0.05,
    window   = 1.12,
    door     = 1.05,
)
    

class MultiarrayDoaDataset(Dataset):
    def __init__(self, 
                 pickle_file, 
                 points,
                 out_dim, 
                 filter_array_idx, 
                 b_divide_by_z = True,
                 testset_split = None):
        """
        b_divide_by_z    bool    [x,y]/z instead of [x,y,z] doa vectors
        """
        # set first calibration point as origin # should only change the output bias / r_0
        b_offset_xy   = False 
        # center doa measures by their means (linear model does not have biases for each doa component)
        b_center_doa  = True 
                 
        self.pickle_file = pickle_file
        data = pickle.load(open(self.pickle_file,'rb'))
        self.out_dim = out_dim
        points = np.array(points,dtype=int)
        self.N_channels = 3
        
        if '29' in pickle_file: # loads all channels
            iin = 0.025
            r = 45/2*iin
            l = 3.07
            self.room = np.zeros((9,3))
            self.room[:,:2] = np.cumsum(np.array([ [-l/2-1.72,-r-1.065], [0,3.63], [1.38,0], [0,.32], [2.33,0], [0,-.6], [6.55-1.38-2.33,0], [0,-3.35], [-6.55,0] ]),axis=0)
            
            self.calib_keys = ['cp{:d}'.format(ii) for ii in range(11)]
            p0  = np.array([      0.00,     0.00, ZREF['table']])
            p1  = np.array([      0.00, -r+5*iin, ZREF['table']])
            p2  = np.array([ -29.5*iin, -r+5*iin, ZREF['table']])
            p3  = np.array([ -49.5*iin, -r+5*iin, ZREF['table']])
            p4  = np.array([-l/2+5*iin,     0.00, ZREF['table']])
            p5  = np.array([ -49.5*iin,  r-5*iin, ZREF['table']])
            p6  = np.array([ -29.5*iin,  r-5*iin, ZREF['table']])
            p7  = np.array([      0.00,  r-5*iin, ZREF['table']])
            p8  = np.array([ 29.75*iin,  r-5*iin, ZREF['table']]) 
            p9  = np.array([ l/2-5*iin,     0.00, ZREF['table']]) 
            p10 = np.array([ 30.75*iin, -r+5*iin, ZREF['table']])
            plist = [p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10]
            
            # include other calibration points
            self.calib_keys.append('ground')
            plist.append(self.room[5]+np.array([.3,0,ZREF['ground']]))
            self.calib_keys.append('door')
            plist.append(self.room[0]+np.array([1.58,0,ZREF['door']]))
            self.calib_keys.append('window')
            plist.append(self.room[0]+np.array([0,0,ZREF['window']]))
            self.calib_keys.append('CL')
            plist.append(self.room[3]+np.array([11.25*iin,-14*iin,ZREF['counter']]))
            self.calib_keys.append('CR')
            plist.append(self.room[4]+np.array([-11.25*iin,-14*iin,ZREF['counter']]))
            self.calib_keys.append('ML')
            plist.append(p5+np.array([0,-14*iin,0]))
            self.calib_keys.append('MR')
            plist.append(p3+np.array([0,+14*iin,0]))
            
            self.calib_xy = np.vstack(plist).astype(np.float32)
            
            # self.arrays = [aa for aa in range(0,7)]
            # [self.arrays.remove(aa-1) for aa in filter_array_idx];
            self.arrays = np.arange(1,8-len(filter_array_idx))
            
            doalist = [get_measurement_array_from_dict(data[key]) for key in self.calib_keys]
            # truncate point measurements
            idx = self.calib_keys.index('cp0')
            doalist[idx] = delft(doalist[idx],2140,2200)
            doalist[idx] = delft(doalist[idx],1070,1090)
            doalist[idx] = delft(doalist[idx],0,20)
            idx = self.calib_keys.index('cp1')
            doalist[idx] = delft(doalist[idx],0,100)
            idx = self.calib_keys.index('cp2')
            doalist[idx] = delft(doalist[idx],0,100)
            doalist[idx] = delft(doalist[idx],-150,None)
            idx = self.calib_keys.index('cp3')
            doalist[idx] = delft(doalist[idx],0,100)
            idx = self.calib_keys.index('cp4')
            doalist[idx] = delft(doalist[idx],1050,1250)
            doalist[idx] = delft(doalist[idx],0,100)
            idx = self.calib_keys.index('cp5')
            doalist[idx] = delft(doalist[idx],0,100)
            idx = self.calib_keys.index('cp6')
            doalist[idx] = delft(doalist[idx],-150,None)
            doalist[idx] = delft(doalist[idx],1080,1100)
            doalist[idx] = delft(doalist[idx],0,250)
            idx = self.calib_keys.index('cp7')
            doalist[idx] = delft(doalist[idx],-100,None)
            doalist[idx] = delft(doalist[idx],0,50)
            idx = self.calib_keys.index('cp8')
            doalist[idx] = delft(doalist[idx],0,20)
            doalist[idx] = delft(doalist[idx],-250,None)
            idx = self.calib_keys.index('cp9')
            idx = self.calib_keys.index('cp10')
            doalist[idx] = delft(doalist[idx],920,1020)
            doalist[idx] = delft(doalist[idx],-100,None)
            
            if 'ground' in self.calib_keys: 
                idx = self.calib_keys.index('ground')
                doalist[idx] = delft(doalist[idx],1600,None)
                doalist[idx] = delft(doalist[idx],0,1000)
            if 'door' in self.calib_keys: 
                idx = self.calib_keys.index('door')
                doalist[idx] = delft(doalist[idx],0,200)
                doalist[idx] = delft(doalist[idx],-250,None)
            if 'window' in self.calib_keys: 
                idx = self.calib_keys.index('window')
                doalist[idx] = delft(doalist[idx],0,300)
            if 'CL' in self.calib_keys: 
                idx = self.calib_keys.index('CL')
                doalist[idx] = delft(doalist[idx],1050,1150)
                doalist[idx] = delft(doalist[idx],0,100)
                doalist[idx] = delft(doalist[idx],-200,None)
            if 'CR' in self.calib_keys: 
                idx = self.calib_keys.index('CR')
                doalist[idx] = delft(doalist[idx],0,300)
            if 'ML' in self.calib_keys: 
                idx = self.calib_keys.index('ML')
                doalist[idx] = delft(doalist[idx],480,620)
                doalist[idx] = delft(doalist[idx],0,200)
                doalist[idx] = delft(doalist[idx],-100,None)
            if 'MR' in self.calib_keys: 
                idx = self.calib_keys.index('MR')
                doalist[idx] = delft(doalist[idx],0,350)
                
            doalist = [dd[:testset_split] for dd in doalist]
        
        else:
            print('undefined init for specified file')
            return None
        self.calib_xy = self.calib_xy[:,:self.out_dim]
        
        if len(points):
            doalist=[doalist[pp] for pp in points]
            self.calib_xy=np.array([self.calib_xy[pp] for pp in points])
        self.N_points = len(doalist)
        self.calib_doa = np.vstack(doalist).astype(np.float32)
        self.calib_id = np.hstack([ii*np.ones(cpm.shape[0]) for ii,cpm in enumerate(doalist)]).astype(int)
        
        self.calib_xy_mean = np.mean(self.calib_xy,axis=0)
            
        # offset training data
        if b_offset_xy:
            self.calib_xy_offset = -plist[0]
            self.calib_xy += self.calib_xy_offset
        else:
            self.calib_xy_offset = np.zeros(self.out_dim,dtype=np.float32)
            
        # Drop arrays
        for fi in np.flip(np.sort(filter_array_idx))-1:
            #print(self.calib_doa.shape, fi, fi*3,fi*3+3)
            self.calib_doa = np.delete(self.calib_doa, range(fi*3,fi*3+3), 1)
        
        if b_divide_by_z:
            # [x/z, y/z] instead of [x,y,z] on a unit sphere
            self.N_channels = 2
            self.calib_doa = np.hstack([
                self.calib_doa[:,3*ii:3*ii+2] / self.calib_doa[:,3*ii+2,np.newaxis] for ii in range(int(self.calib_doa.shape[1]/3)) ])
            self.calib_doa = np.nan_to_num(self.calib_doa, copy=False)
            
            
        self.N_arrays = int(self.calib_doa.shape[1]/self.N_channels)
        
        self.calib_doa_mean   = np.mean(self.calib_doa, axis=0)
        if b_center_doa: # offset DOA by -mean 
            self.calib_doa_offset = -self.calib_doa_mean
            self.calib_doa += self.calib_doa_offset
        else:
            self.calib_doa_offset = np.zeros(self.calib_doa.shape[1])
        
        # load clean DOA data from arrays [0,1,2,3,5]
        # self.test_data = data["active_L_table_slide_DOA"]
        # _matrix includes timestamp (first row) and array#4 (all_nans)
        # self.active_L_table_slide_matrix = data["active_L_table_slide_matrix"]
        
        
        #self.active_long_table_slide_DOA = data["active_long_table_slide_DOA"]
        #self.active_long_table_slide_matrix = data["active_long_table_slide_matrix"]
        
    def __len__(self):
        return len(self.calib_id)
    
    def xyi(self,idx):
        return self.calib_xy[idx] # only obtain 2D calibration coordinates
    
    def __getitem__(self,idx):
        return self.calib_doa[idx,:], self.xyi(self.calib_id[idx])

def get_test_data(filepath,out_dim,remove_array_idx=[],test_keys=None,doa_offset=0,b_divide_by_z=True,
                 testset_split=0,# to split DOAs across training and test set
                 ):# load and format test data
    data = pickle.load(open(filepath,'rb'))
    test_cases = list()
    test_zref = list()
    
    if test_keys is None:
        test_keys =  [
            # included
            'rectangle', 
            'inner', 
            'counter', #trace

            # # cleaned but not included
            # 'ground', # point, z = floor
            # 'window', #point
            # 'door', # point
            # 'outer', 
            # 'table1', # smooth
            # 'MR', # suspect monitor points
            # 'ML', 

            # neither clean nor included
            # 'table4', 'table2', 'table3',  'table5',  'table6', # disturbed
            # 'TR', 'TL', 'L2R', 'R2L', 
            # 'CL', 'CR', 'cR2L', 'cL2R',  
        ]
    for key in test_keys:
        test_data = get_measurement_array_from_dict(data[key]).astype(np.float32)

        # trim/clean specific point measurements
        #idx = test_keys.index(key)
        if key == 'inner': 
            test_data = delft(test_data,988,990)
            test_data = delft(test_data,0,50)
        elif key == 'counter': 
            test_data = delft(test_data,0,300)
        elif key == 'door': 
            test_data = delft(test_data,0,200)
            test_data = delft(test_data,-250,None)
            test_data = delft(test_data,0,testset_split)
        elif key == 'window': 
            test_data = delft(test_data,0,300)
            test_data = delft(test_data,0,testset_split)
        elif key == 'ground': 
            test_data = delft(test_data,0,1000)
            test_data = delft(test_data,1600,None)
            test_data = delft(test_data,0,testset_split)
        elif key == 'ML': 
            test_data = delft(test_data,480,620)
            test_data = delft(test_data,-200,None)
            test_data = delft(test_data,0,200)
        elif key == 'MR': 
            test_data = delft(test_data,0,350)
        elif key == 'table1': 
            test_data = delft(test_data,0,180)
            test_data = delft(test_data,1540,None)
        elif key == 'CL': 
            test_data = delft(test_data,1050,1150)
            test_data = delft(test_data,0,100)
            test_data = delft(test_data,-200,None)
        elif key == 'CR': 
            test_data = delft(test_data,0,300)
        if key in ['table1','ML','MR','rectangle','inner','outer']:
            test_zref.append(ZREF['table']*np.ones(len(test_data)))
        else: 
            test_zref.append(ZREF[key]*np.ones(len(test_data)))

        # replace labels
        test_labels = test_keys.copy()
        if 'inner' in test_labels: test_labels[test_labels.index('inner')]='screen'

        for fi in np.flip(np.sort(remove_array_idx))-1:
                test_data = np.delete(test_data,range(3*fi,3*fi+3),1)
        if b_divide_by_z:
            test_data = np.hstack([
                test_data[:,3*ii:3*ii+2] / test_data[:,3*ii+2,np.newaxis] 
                for ii in range(int(test_data.shape[1]/3))
            ])
            np.nan_to_num(test_data, copy=False)

        test_data += doa_offset # iff offset in training data, apply here too
        test_data = torch.from_numpy(test_data.astype(np.float32))
        test_cases.append(test_data)
        
    return test_cases, test_labels, np.hstack(test_zref)


def plot_room(ax, room, color_refs=False):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    pargs = {'color': '#dddddd','zorder':0.9,'alpha':1.0}
    text_opts = {'fontsize':18}
    d = 1.185
    r = d/2
    l = 3.07
    ax.plot(room[:,0],room[:,1],'k',lw=2)
    alpha = np.linspace(np.pi,3/2*np.pi,100)
    cake = np.array([np.cos(alpha), np.sin(alpha)]).T
    door = room[6] + np.array([[-.2,0]]) + np.vstack([[0,0], cake,[0,0]])
    ax.plot(door[:,0],door[:,1],'k',alpha=.5)
    furniture = [
        mpl.patches.Rectangle(tuple(room[3]-[0,.6]),2.33,.6, **pargs), # counter
        mpl.patches.Rectangle((-1.535+r,-r),3.07-d,d, **pargs), # table
        mpl.patches.Ellipse(  (-1.535+r,0),d,d,**pargs),
        mpl.patches.Ellipse(  ( 1.535-r,0),d,d,**pargs),
    ]
    screen_center = np.array([-l/2+5*.025,     0.00])
    ax.plot(screen_center[0]+[14*.025,14*.025],screen_center[1]+[-.30,.30],'k',alpha=.7,lw=4,zorder=2)
    [ax.add_patch(ff) for ff in furniture]
    
    if color_refs:
        ref_color=None
        ref_labels=['rectangle','screen','counter']
        ref_alpha = 1.0
    else:
        ref_color='k'
        ref_labels=[None,None,None]
        ref_alpha = 0.3
    iin = .025
    p0  = np.array([      0.00,     0.00, 0.74])
    p1  = np.array([      0.00, -r+5*iin, 0.74])
    p2  = np.array([ -29.5*iin, -r+5*iin, 0.74])
    p3  = np.array([ -49.5*iin, -r+5*iin, 0.74])
    p4  = np.array([-l/2+5*iin,     0.00, 0.74])
    p5  = np.array([ -49.5*iin,  r-5*iin, 0.74])
    p6  = np.array([ -29.5*iin,  r-5*iin, 0.74])
    p7  = np.array([      0.00,  r-5*iin, 0.74])
    p8  = np.array([ 29.75*iin,  r-5*iin, 0.74]) 
    p9  = np.array([ l/2-5*iin,     0.00, 0.74]) 
    p10 = np.array([ 30.75*iin, -r+5*iin, 0.74])
    plist = np.array([p2,p6,p8,p10,p2])
    ax.plot(plist[:,0],plist[:,1],'--',c=ref_color,alpha=ref_alpha,label=ref_labels[0],lw=2)
    alpha = np.linspace(3/2*np.pi,1/2*np.pi,100)
    angle = np.array([np.cos(alpha), np.sin(alpha)])
    radius= (r-5**iin)*(2+np.cos(alpha))/2
    cake = -angle*radius
    plist = (p3+(p5-p3)/2)[:2] +np.vstack([[0,0], cake.T,[0,0]])
    ax.plot(plist[:,0],plist[:,1],'--',c=ref_color,alpha=ref_alpha,label=ref_labels[1],lw=2)
    #cl = training_data.room[3]+np.array([11.25*iin,-14*iin,0])
    #cr = training_data.room[4]+np.array([-11.25*iin,-14*iin,0])
    cl = np.array([-1.59375,  1.9725 ,  0.     ])
    cr = np.array([0.17375 ,1.9725 , 0.     ])
    off1 = np.array([-2*iin,-2*iin,0])
    off2 = np.array([-2*iin, 2*iin,0])
    plist = np.array([cl+off1,cl+off2,cr-off1,cr-off2,cl+off1])
    ax.plot(plist[:,0],plist[:,1],'--',c=ref_color,alpha=ref_alpha,label=ref_labels[2],lw=2)
    
    ax.set_aspect('equal')
    ax.set_anchor('SW')
    ax.set_xlabel("$x$ [m]",**text_opts)
    ax.set_ylabel("$y$ [m]",**text_opts)
   
    xl = ax.get_xlim()
    xl = np.array([np.floor(xl[0]), np.ceil(xl[1])])
    yl = ax.get_ylim()
    yl = np.array([np.floor(yl[0]), np.ceil(yl[1])])
    xt = np.arange(*xl*2)/2
    yt = np.arange(*yl*2)/2
    ax.set_xticks(xt,minor=True)
    ax.set_yticks(yt,minor=True)
    xt = np.arange(*xl)
    yt = np.arange(*yl)
    ax.set_xticks(xt,minor=False)
    ax.set_yticks(yt,minor=False)
    ax.grid(True,which='both',zorder=2)
    
    
def show_localization(pred, training_data, color=None,label=[""], r0 = None,ax=None,ylims=None,xlims=None,lopts=dict(),popts=dict(),title=None,color_refs=False):
    """plot localization prediction within the room"""
    import matplotlib.pyplot as plt
    opts = dict(zorder=3)
    legend_opts = {'ncol':1,'loc':'upper right','bbox_to_anchor':(1.18,1.0),'handletextpad':.1}
    legend_opts.update(lopts)
    calibopts = dict(edgecolors='k',marker='o',facecolors='none', s=100, zorder=2, lw=2)
    if ax == None:
        fig = plt.figure(figsize=(5,5))
        ax = plt.subplot(111)
        
    
    if type(pred) is not list: pred = [pred]
        
    # figure out if separate list elements should have same color and only one label
    everynth = int(len(pred)/len(label))
    if everynth <1: everynth = 1
    for tt,pp in enumerate(pred):
        if color is None: 
            if len(pred)>len(label):
                opts.update({'c':'C{:d}'.format(int(tt/everynth))})
            else:
                opts.update({'c':f'C{tt:d}'})
        elif type(color) is list:  
            if len(pred)>len(color):
                opts.update({'c':color[int(tt/len(color))]})
            else:
                opts.update({'c':color[tt]})
        else:              
            opts.update({'c':color})

        if type(label) is list:
            if (len(pred)>len(label)) & (int(tt%everynth) != 0):
                opts.update({'label': None})
            else:
                opts.update({'label':label[int(tt/everynth)]})
        
        if color_refs:
            opts.update({'marker':'x','s':100})
            opts.update(popts)
            ax.scatter(pp[:,0],pp[:,1],**opts)
            legend_opts.update({'markerscale':1.2})
            calibopts.update(dict(label='ref.'))
        else:
            opts.update(popts)
            ax.plot(pp[:,0],pp[:,1],**opts)
            
    plot_room(ax,room = training_data.room[:,:2],color_refs=color_refs)
    ax.scatter(training_data.calib_xy[:,0],training_data.calib_xy[:,1],  **calibopts)
    if ylims: ax.set_ylim(ylims)
    if xlims: ax.set_xlim(xlims)
    if not label==[""]: ax.legend(**legend_opts,fontsize=18)
    if title is not None: plt.text(-.03, 1.05, title, transform=ax.transAxes, fontsize=20)
    return ax


def affine_transform(D, R, rcond=None, trunc=None):
    """
    # D: list of DOA arrays for each point
    # R: coordinates for each point
    # optional:
        # trunc: -> number of PCA components to be considered
        # rcond: float within 0..1 -> limit below which normalized singular values are truncated
    
    # B
    """
    R_mean = np.mean(R, axis=1).reshape(-1,1)
    D_mean = np.mean(D, axis=1).reshape(-1,1)
    
    # calculating moore penrose inverse
    # via SVD
    ### assembled D = (U@np.diag(sigma))@VH[:len(sigma)]
    ### truncated D = (U[:,;trunc]@np.diag(sigma[:trunc]))@VH[:trunc]
    ### MP inv    D = (VH[:trunc].T@np.diag(1/sigma[:trunc]))@U.T[:trunc]
    
    #U,sigma,VH = np.linalg.svd(D-D_mean)
    U,sigma,VH = np.linalg.svd((D-D_mean)@D.T)
    sscaled = sigma/sigma[0]
    print('(D-D_mean)@D^T\n - singular values (scaled):',sscaled, '\n - condition number:',sigma[0]/sigma[-1])
    if not trunc:
        if rcond: 
            trunc = np.argwhere(sscaled<rcond)[0][0]
    if trunc:
        print(' - kept singular values' , sscaled[:trunc], '\n - new condition number:',sigma[0]/sigma[trunc-1])
        D_inv = (VH[:trunc].T@np.diag(1/sigma[:trunc]))@U.T[:trunc]
    else:
        print(' - keep all singular values, no truncation')
        D_inv = (VH[:len(sigma)].T@np.diag(1/sigma))@U.T
    
    ### via rcond directly 
    # if not rcond: 
    #     if trunc:
    #         rcond = sscaled[trunc-1]*.99
    #     else:
    #         rcond = 1e-3 # as good as full
    # D_inv = np.linalg.pinv(D-D_mean, rcond = rcond) 
    
    # obtaining linear transformation matrix
    # B = (R - R_mean) @ D_inv 
    B = ((R - R_mean)@D.T) @ D_inv 
    #B2 = (R-R_mean) @ np.linalg.pinv(D-D_mean)
    #B3 = ((R-R_mean)@D.T) @ np.linalg.pinv((D-D_mean)@D.T)
    R_0 = R_mean - B @ D_mean
    return B, R_0, U, sigma , VH

def plot_singular_values(sv,LM_TRUNC,ax=None,title="Affine mapping regularization"):
    import matplotlib.pyplot as plt
    if ax is None: 
        fig=plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
    
    jj = np.arange(1,len(sv)+1)
    ax.stem(jj,sv,linefmt='k',markerfmt='ko')
    #ax.set_yscale("log")
    ax.set_xticks(jj)
    ax.axvline(LM_TRUNC+.5,linestyle='dashed',c='r')
    ax.set_ylabel('$j^{th}$ singular value, $\Lambda_{jj}\,/\,\Lambda_{11}$')
    ax.set_xlabel('$j$')
    ax.text(-.03,1.05,title,transform=ax.transAxes,fontsize=20)
    ax.grid(True)