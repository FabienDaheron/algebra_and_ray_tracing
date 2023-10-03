""" @author: DAHERON FABIEN """
# TO DO: Typing verification
# TO DO: Test function

#=============================================================================
# --------- Import libraries ---------
#=============================================================================
import numpy as np
from typing import Union
from reference_frame import ReferenceFrame, cartesian_frame
from vector import Vector, cartesian_basis
import copy


#=============================================================================
# --------- Class Point ---------
#=============================================================================
class Point : 
    """
    Introducing
    ----------
        A point of R^N is defined by its coordinates in a giving reference 
        frame.
            
        The cartesian vectors of R^N are noted fj for j in [1,N].
        We recall that fj = [0.0
                             0.0
                             ...
                             1.0 coefficient number j
                             ...
                             0.0]
        
        The basis vectors are noted ei for i in [1,N].
        We have ei = [ei1
                      ei2
                      ...
                      eij coefficient number j
                      ...
                      eiN]
                     
        Such as ei = ei1*f1 + ei2*f2 + ... + eij*fj + ... + eiN*fN.
        
        The origin of the frame is noted o.
        We have o = [o1
                     o2
                     ...
                     oj coefficient number j
                     ...
                     oN]
        
        Such as o = o1*f1 + o2*f2 + ... + oj*fj + ... + oN*fN.
        
        The point is noted p.
        We have p = [p1
                     p2
                     ...
                     pj coefficient number j
                     ...
                     pN]
        
        Such as p = p1*e1 + p2*e2 + ... + pj*ej + ... + pN*eN.
        
        However, the type 'Point' allows to manipulate M points at the same 
        time. Fot it, the M points needs to be expressed in the same basis. 
        The M points are defined by their coordinates in a giving frame.
        The M points are noted vi for k in [1,M].
        Lets consider the matrix m with size N*M
        
        m = [p1 p2 ... pk ... pM] = [p11 p21 ... pk1 ... pM1
                                      p12 p22 ... pk2 ... pM2
                                      ... ... ... ... ... ...
                                      p1j p2j ... pkj ... pMj
                                      ... ... ... ... ... ...
                                      p1N p2N ... pkN ... pMN]
        
        Such as pk = pk1*e1 + pk2*e2 + ... + pkj*ej + ... + pkN*eN.
        
        We construct also mc the matrixe vector expressed in the cartesian 
        frame of reference.
        
    Parameters
    -----------
        frame: ReferenceFrame
               The frame {o,e1,e2, ..., eN} in witch the points are defined.
               
        *P: np.ndarray with shape (N,) or list or tuple with len N 
            The pi point expressed in the given frame for i in [1,M].
            P = (p1,p2 ... pM)
            {M pi are requiered}
            
    Attributes
    ----------      
        frame: ReferenceFrame
               The frame {o,e1,e2, ..., eN} in witch the points are defined
     
        m: np.ndarray with shape (N,M)
            The coordinates af the M points in the given frame.
       
        mc: np.ndarray with shape (N,M)
             The coordinates af the M points in the cartesian frame of 
             reference.
                               
        N: int
           The dimension of the vector space.
           
        M: int
           The number of points contained in the object.
        
    Using
    ----------
        We want to create 2 points p1 and p2 of dimension 3 in a given frame 
        named 'frame'.
        p1 = [1        p2 = [1
              2              0
              1]             0]
            
        A solution is : 
            > p1 = np.array([1,2,1]) or [1,2,1] or (1,2,1)
            > p2 = np.array([1,0,0])
            > point = Point(frame,p1,p2)
            
        A other solution is :
            > P = [ [1,2,1] , [1,2,1]]
            > point = Point(frame,*P)
        
        To change realize a change of frame, use :
            > point.frame = new_frame
        
        If you have the matrix m or mc and you want to create a new Point :
        You can use : 
            > new_point = Point(frame, np.zeros((frame.N,)))
            > new_point.m = m
            
    Notes
    ----------
        Be carreful : pij = point.m[j-1,i-1]
    
        Do not modify coefficients of 'm' or 'mc' directly.
        Use 'set_point', 'set_point_coef' methods otherwise 'm' and 'mc' won't
        be in correspondance.
        
        You can also set 'm' and 'mc' entirely. 
        
        If you had modify 'm' or 'mc' coefficients 
        with an other way, use 'update_m' or 'update_mc' to correct the
        correspondance.              
    """
    def __init__(self, frame: ReferenceFrame,
                 *P: Union[np.ndarray,list,tuple]) -> None:
        self.__frame = copy.copy(frame)
        #coordinates of the vector
        for pi in P :
            pi = np.array(pi).astype(float).reshape((self.__frame.N,))
        self.__m = np.transpose(list(P)).astype(float)
        self.update_mc()
        
    #======================================================================
    # --------- Setter and Getter methods  ---------
    #======================================================================    
    @property
    def frame(self) -> ReferenceFrame:
        """Returns the frame in witch the points are defined"""
        return self.__frame
    
    @frame.setter
    def frame(self, frame: ReferenceFrame) -> None:
        """
        Changes the frame of the object. 
        Realizes the change of frame to expressed the matrix m in this new 
        frame.
        
        Parameters
        -----------
            frame: ReferenceFrame
                   The frame {o,e1,e2, ..., eN} in witch the points are 
                   defined.
        """
        self.__frame = copy.copy(frame)
        self.__m = self.__frame.cart2frame(self.__mc)
        
    @property
    def N(self) -> int:
        """Returns the dimension of the R^N vector space."""
        return self.__m.shape[0]
        
    @property
    def M(self) -> int:
        """Returns the number of points in the object"""
        return self.__m.shape[1]
    
    @property
    def m(self) -> np.ndarray:
        """Returns the matrix corresponding to the points coordinates in the 
        given frame"""
        return self.__m
    
    @m.setter
    def m(self, m: np.ndarray) -> None:
        """
        Sets the matrix corresponding to the points coordinates in the 
        given frame = self.frame.
        
        If you want to redefine m in a new frame, many solutions : 
            - change the frame and after set the matrix m
                > point.frame = new_frame
                > point.m = new_m {expressed in the new frame}
            - use the method 'set_m'
                > point.set_m(new_frame, new_m)
                
        If you want to change only some coefficients of m please check methods
        'set_point' and 'set_point_coef' but do not change the matrix directly! 
        
        If you need to change directly a coefficent, please use the method 
        update_mc after. {not recommended}
        
        See also add_points and del_points to add or del points in the objet.
                
        Parameters
        -----------
            m: np.ndarray with shape (N,M)
                The coordinates af the M points in the given frame.
        """
        self.__m = np.copy(m).astype(float)
        self.update_mc()
        
    def set_m(self, frame: ReferenceFrame, m: np.ndarray) -> None:
        """
        Sets the matrix corresponding to the points coordinates in an new 
        given frame.
                
        Parameters
        -----------
            frame: ReferenceFrame
                   The frame {o,e1,e2, ..., eN} in witch the points are 
                   defined.
        
            m: np.ndarray with shape (N,M)
                The coordinates af the M points in the given frame.
        """
        self.__frame = copy.copy(frame)
        self.__m = np.copy(m).astype(float)
        self.update_mc()
    
    @property
    def mc(self) -> np.ndarray:
        """Returns the matrix corresponding to the points coordinates in the 
        cartesian frame of reference"""
        return self.__mc
    
    @mc.setter
    def mc(self, mc: np.ndarray) -> None:
        """
        Sets the matrix corresponding to the points coordinates in the 
        cartesian frame of reference.
        
        If you want to change only some coefficients of mc please check 
        methods 'set_point' and 'set_point_coef' but do not change the matrix 
        directly! 
        
        If you need to change directly a coefficent, please use the method 
        update_m after. {not recommended}
        
        See also add_points and del_points to add or del points in the
        objet.
        
        Parameters
        -----------
            mc: np.ndarray with shape (N,M)
                The coordinates af the M points in the cartesian frame of 
                reference.
        """
        self.__mc = np.copy(mc).astype(float)
        self.update_m()
    
    def set_point(self, i: int, frame: ReferenceFrame, pi: np.ndarray) -> None:
        """
        Sets the point pi with an new np.ndarray with shape (N,).
        Coordinates of the point pi is expressed in the given frame but will
        be converted into the frame self.frame.
        
        Parameters
        -----------
            i: int
               The point pi to modify, i in [1,M].
               
            frame: ReferenceFrame
                   The frame {o,e1,e2, ..., eN} in witch the point is defined.
            
            pi: np.ndarray with shape (N,) or list or tuple with len N
                The pi point expressed in the given frame.
        """
        #coordinates of the point in the cartesian frame (cvi) and the
        #self.frame (vi)
        cpi = frame.frame2cart(np.array(pi).astype(float).reshape((self.N,))) 
        pi = self.__frame.frame2cart(cpi) #self.frame frame
        #Modification of m and mc
        self.__m[:,i-1] = pi
        self.__mc[:,i-1] = cpi
    
    def set_point_coef(self, i: int, j: int,
                        frame: ReferenceFrame, value: float) -> None:
        """
        Sets the coordinate j of the point pi with an new float value.
        The value correspond of the pi.ej coordinate in the given frame 
        but will be converted into the frame self.frame.
        
        Parameters
        -----------
            i: int
               The point pi to modify, i in [1,M].
               
            j: int
               The coefficient j of pi to modify, j in [1,N].
               
            frame: ReferenceFrame
                   The frame {o,e1,e2, ..., eN} in witch the point is defined.
            
            value: float
                   The value to set at the pij coefficient.
        """
        #point pi in the given frame
        pi = frame.cart2frame(self.mc[:,i-1])
        pi[j-1] = float(value)
        #modification of v
        self.set_point(i, frame, pi)
        
    #======================================================================
    # --------- overwritting methods ---------
    #======================================================================
    def __repr__(self) -> str:
        """ Returns the representation of the point."""
        s: str = f'{self.__class__.__name__} : N = {self.N} ; M = {self.M}' 
        return s
    
    def __str__(self) -> str:
        """ Returns the string describing the point."""
        s: str = (f'{self.__class__.__name__} : N = {self.N} ; M = {self.M}' 
                  + '\n-----------------')
        if self.N <= 3:
            #first case : N <= 3 and M <= 3
            if self.M <= 3:
                for i in range(0,self.M):
                    s = s + f'\nv{i+1} = ['
                    for j in range(0,self.N):
                        s = s + f'{self.mc[j,i]:>8.2f} '
                    s = s + ']'
            #second case : N <= 3 and M > 3
            else:
                for i in [0,1]:
                    s = s + f'\nv{i+1} = ['
                    for j in range(0,self.N):
                        s = s + f'{self.mc[j,i]:>8.2f} '
                    s = s + ']'
                s = s  + '\n  ... '
                i = self.M-1
                s = s + '\nvM = ['
                for j in range(0,self.N):
                    s = s + f'{self.mc[j,i]:>8.2f} '
                s = s + ']'
        else:  
            #third case : N > 3 and M <= 3
            if self.M <= 3:
                for i in range(0,self.M):
                    s = s + f'\nv{i+1} = ['
                    for j in [0,1]:
                        s = s + f'{self.mc[j,i]:>8.2f}'
                    s = s + ' ... '
                    s = s + f'{self.mc[self.N-1,i]:>8.2f}]'
            #last case : N > 3 and M > 3
            else: 
                s = s + (f'\nv{1} = [{self.mc[0,0]:>8.2f} '
                      + f'{self.mc[1,0]:>8.2f} ... '
                      + f'{self.mc[self.N-1,0]:>8.2f} ]'
                      + f'\nv{2} = [{self.mc[0,1]:>8.2f} '
                      + f'{self.mc[1,1]:>8.2f} ... '
                      + f'{self.mc[self.N-1,1]:>8.2f} ]'
                      + '\n  ... '
                      + f'\nvM = [{self.mc[0,self.M-1]:>8.2f} '
                      + f'{self.mc[1,self.M-1]:>8.2f} ... '
                      + f'{self.mc[self.N-1,self.M-1]:>8.2f} ]')
        return s
    #======================================================================
    # --------- methods ---------
    #====================================================================== 
    def update_m(self) -> None:
        """Updates m after a modification of mc."""
        self.__m = self.__frame.cart2frame(self.__mc)
        
    def update_mc(self) -> None:
        """Updates mc after a modification of m."""
        self.__mc = self.__frame.frame2cart(self.__m)
        
    def add_points(self, frame: ReferenceFrame,
                    *P: Union[np.ndarray,list,tuple]) -> None:
        """
        Adds p new points in the object vector. 
        The points are expressed in the given frame but will be convert into 
        the frame self.frame.
        
        The new points are noted nP = (np1,np2, ..., npp) such as the 
        new M attributs of is object become M + p. 
        
        The new points are placed after current points, such as the points
        P in the objet are :
        P = (p1,p2,...,pM,pM+1,...,pM+P) = (p1,p2,...,pM,np1,np2,...,npp)
            
        Parameters
        -----------
            frame: ReferenceFrame
                   The frame {o,e1,e2, ..., eN} in witch the points is defined.
                   
            *P: np.ndarray with shape (N,) or list or tuple with len N 
                The npi new points expressed in the given frame for i in [1,p].
                nP = (np1,np2, ..., npp)
                {P pi are requiered}
        """
        #construction of the m and the mc matrix of new point
        for pi in P :
            pi = np.array(pi).astype(float).reshape((self.__frame.N,))
        m_frame = np.transpose(list(P)).astype(float) #is the given frame
        mc = frame.frame2cart(m_frame) #in the cartesian frame
        m = self.__frame.cart2frame(mc) #in the frame of self
        #Concatenation of self.mc and mc, and concatenation of self.m and m
        self.__m = np.concatenate((self.__m,m),axis=1)
        self.__mc =  np.concatenate((self.__mc,mc),axis=1)
        
    def del_points(self, bool_array: Union[list,tuple,np.ndarray]) -> None:
        """
        Removes some points in the object.
        
        bool_array is an array of shape (M,) witch contains M booleans. 
        points corresponding to a False are removed.
            
        Parameters
        -----------
            bool_array: np.ndarray with shape (M,) or list or tuple with len M
                        The array or list containing M booleans (or 0 and 1). 
                        Only the points corresponding to False values are 
                        removed.
        """
        #construction of the m and the mc matrix of new point
        bool_array = np.array(bool_array).astype(bool).reshape(self.M,)
        self.__m = self.__m[:,bool_array]
        self.__mc = self.__mc[:,bool_array]    
        
    def vectorAB(self, point: 'Point') -> Vector:
         """
         Return the vector AB from self to point.
         
         if point.M = 1:
             -> returns the the vector between self and the point.
             
             self : P = (p1,p2,...,pM)
             point : U = (u1)
             
             return : PU = (u1 - p1,u1 - p2,...,u1 - pM)
         
         if point.M = 1:
             -> returns the the vector between each point of self and
             the correspondinf point in point.
             
             self : P = (p1,p2,...,pM)
             point : U = (u1,u2,...,uM)
             
             return : PU = (u1 - p1,u2 - p2,...,uM - pM)
         
         Parameters
         -----------
             point: Point
                    The point B of the vector AB.
                    
        Returns
        ----------
            vector: Vector
                    The vector AB between self and point.
         """
         V = Vector(cartesian_basis(self.N),np.zeros((self.N,)))
         V.mc = point.mc - self.mc
         return V 

#=============================================================================
# --------- Commun Vectors ---------
#=============================================================================
def randn_point(N: int, M: int, frame: ReferenceFrame) -> Vector :
    """
    Returns M randn point of R^N defined in the frame frame.
    
    Parameters
    ----------
        N: int
           The dimension of the vector space R^N.
           
        M: int 
           The number of points in the point objet.
           
        frame : ReferenceFrame
                The frame {o,e1,e2, ..., eN} in witch the points is defined.
    """
    P = []
    for k in range(M):
        P.append(np.random.randn(N))
    return Point(frame, *P)        

#==============================================================================
# --------- Test ---------
#==============================================================================
if __name__ == '__main__' : 
    N = 3
    M = 10000
    point = randn_point(N, M, cartesian_frame(N))
    print('point \n',point,'\n')
    
    
