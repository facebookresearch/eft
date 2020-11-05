# Original code from SPIN: https://github.com/nkolot/SPIN

import numpy as np
import scipy

#To conversion between densepose output to SMPL 
class denseposeManager():

    def __init__(self, width=1600, height=1200, name='GL Renderer',
                 program_files=['renderer/shaders/simple140.fs', 'renderer/shaders/simple140.vs'], color_size=1, ms_rate=1):
        """
        Initialize the engine.

        Args:
            self: (todo): write your description
            width: (int): write your description
            height: (int): write your description
            name: (str): write your description
            program_files: (str): write your description
            color_size: (int): write your description
            ms_rate: (float): write your description
        """
        
        
        self.densepose_info = self.loadDensepose_info()

        #Densepose Specific 
        self.dp_faces = self.densepose_info['All_Faces']-1 #0~13774
        self.dp_vertexIndices = self.densepose_info['All_vertices']-1    #(1,7829)       #Vertex orders used in denpose info. There are repeated vetices
        #### self.dp_vertexIndices has 0-6889


        #DP color information
        dp_face_seg = self.densepose_info['All_FaceIndices']     #(13774,1)     #Seg infor for face

        dp_vertex_seg = np.zeros( (7829,1))     #(7829,1)

        dp_vertex_seg[self.dp_faces[:,0]] = dp_face_seg           #Overwrite if repeated
        dp_vertex_seg[self.dp_faces[:,1]] = dp_face_seg
        dp_vertex_seg[self.dp_faces[:,2]] = dp_face_seg
        assert(np.min(dp_vertex_seg) >0)        #dp_vertex_seg: 1~24

        dp_vertex_U = self.densepose_info['All_U_norm']     #(7289,1)

        dp_vertex_V = self.densepose_info['All_V_norm']     #(7829,1)

        """
        Build UVI map in SMPL vertex order
        smpl_vertex_Seg[veridx_smpl]     #6890,1
        smpl_vertex_U[veridx_smpl]       #6890,1
        smpl_vertex_V[veridx_smpl]       #6890,1
        smpl_IUV[veridx_smpl] --> i,u,v for each verex  #6890,3

        iuv_to_smplvertex[(i,u,v)] --> smpl_idx  #no barycentric. Ignor if missing. I,U,V are integers
        """


        #Setupt IUV to smpl vertexId
        #vertexidx_smplorder = self.dp_vertexIndices[vertexidx_dporder] 
        #ver_dporder = ver_smplorder[self.dp_vertexIndices]       (7829, 3)
        smpl_vertex_U = np.zeros( (6890,1) )
        smpl_vertex_U[self.dp_vertexIndices[-1]] = dp_vertex_U

        smpl_vertex_V = np.zeros( (6890,1) )
        smpl_vertex_V[self.dp_vertexIndices] = dp_vertex_V

        smpl_vertex_Seg = np.zeros( (7829,1) )
        smpl_vertex_Seg = np.zeros( (6890,1) )
        smpl_vertex_Seg[self.dp_vertexIndices] = dp_vertex_seg

        self.smpl_vertex_IUV = np.concatenate( [smpl_vertex_Seg,smpl_vertex_U,smpl_vertex_V], axis=1)           #6890,3


        print("Generated SMPL IUV vertex info: {}", self.smpl_vertex_IUV.shape)

        


        


    #make sure you have: /yourpath/bodymocap/renderer/densepose_uv_data/UV_Processed.mat
    def loadDensepose_info(self, dp_data_path= 'renderer/densepose_uv_data/UV_Processed.mat'):
        """
        Loads dense dataset.

        Args:
            self: (todo): write your description
            dp_data_path: (str): write your description
        """
        
        #Load densepose data
        import scipy.io as sio
        densepose_info = None
        densepose_info = sio.loadmat(dp_data_path)      #All_FaceIndices (13774), All_Faces(13774), All_U(7829), All_U_norm(7829), All_V(7829), All_V_norm (7829), All_vertices (7829)
        assert densepose_info is not None
        # All_FaceIndices - part labels for each face
        # All_Faces - vertex indices for each face
        # All_vertices - SMPL vertex IDs for all vertices (note that one SMPL vertex can be shared across parts and thus appear in faces with different part labels)
        # All_U - U coordinates for all vertices
        # All_V - V coordinates for all vertices
        # All_U_norm - normalization factor for U coordinates to map them to [0, 1] interval
        # All_V_norm - normalization factor for V coordinates to map them to [0, 1] interval
        # vertexColor = densepose_info['All_U_norm']*255
        # vertexColor = np.zeros((v.shape[1], 3))
        # vertexColor[:,0] = densepose_info['All_U_norm'][:v.shape[1]].flatten()       #(6890,3)
        # vertexColor[:,1] = densepose_info['All_V_norm'][:v.shape[1]].flatten()       #(6890,3)

        # # faces = smplWrapper.f
        # v =v[0]  #(6890,3)
        # dp_vertex = v[densepose_info['All_vertices']-1]  #(1,7829,3)        #Considering repeatation
        # faces =densepose_info['All_Faces']-1 #0~7828
        # # vertexColor = densepose_info['All_FaceIndices']     #(13774,1)
        # # vertexColor = np.repeat(vertexColor,3,axis=1) /24.0   #(13774,3)

        # # vertexColor = densepose_info['All_U_norm']     #(13774,1)
        # vertexColor = densepose_info['All_V_norm']     #(13774,1)
        # vertexColor =  np.repeat(vertexColor,3,axis=1) 

        # # vertexColor[vertexColor!=2]*=0
        # vertexColor[vertexColor==2]=24
        return densepose_info


    def  conv_iuv_smplIdx(self, dp_iuv_arr):
        """
        Convert IUV array (3,X,Y) to SMPL vertex 
        """
        i_map = dp_iuv_arr[0]
        u_map = dp_iuv_arr[1] /255.0
        v_map = dp_iuv_arr[2]  /255.0

        iuv2ver =[]
        # valididx = i_map >0
        # i_array = i_map[valididx]
        # u_array = u_map[valididx]
        # v_array = v_map[valididx]

        
        for r in range(0, i_map.shape[0],10):
            for c in range(0, i_map.shape[1],10):
                
                i = i_map[r,c]
                if i==0:
                    continue

                u = u_map[r,c]
                v = v_map[r,c]

                verIdx = self.IUV2VertexId( i,u,v)
                # print(f"{c},{r} ( {i}, {u}, {v}) --> {verIdx}")
                iuv2ver.append( np.array([c,r,verIdx]))

        iuv2ver = np.array(iuv2ver)


        #(X,Y)  -> Vertex ID
        return iuv2ver

    
    
    def IUV2VertexId( self, I_point , U_point, V_point):
        """
        Find the vertex between two vertex

        Args:
            self: (todo): write your description
            I_point: (todo): write your description
            U_point: (int): write your description
            V_point: (int): write your description
        """
        P = [ U_point , V_point , 0 ]
        VertexIndicesNow  = np.where( self.smpl_vertex_IUV[:,0] == int(I_point) )[0]
        IUV_sub = self.smpl_vertex_IUV[VertexIndicesNow]        #for selected part's uv only

        #Find closest one
        #
        UV_dist = scipy.spatial.distance.cdist( np.array( [U_point,V_point])[np.newaxis,:] , IUV_sub[:,1:]).squeeze()
        closestVertext = np.argmin(UV_dist)
        vertexIdx = VertexIndicesNow[closestVertext]

        return vertexIdx