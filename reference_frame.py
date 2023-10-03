""" @author: DAHERON FABIEN """
# TO DO: Typing verification
# TO DO: Test function

#=============================================================================
# --------- Import libraries ---------
#=============================================================================
import numpy as np
import random
from typing import Union
from basis import Basis

#=============================================================================
# --------- Class ReferenceFrame ---------
#=============================================================================
class ReferenceFrame(Basis):
    """
    Introducing
    ----------
        A frame of reference of R^N is defined with an origin and a basis.
        The basis is defined with N vectors of dimension N and the origin with
        a vector of dimension N.         
        The basis vectors and the origine vector are defined in the cartesian 
        frame of reference of R^N.
            
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
        
        By the same construction, the origine vector is noted o.
        We have o = [o1
                     o2
                     ...
                     oj coefficient number j
                     ...
                     oN]
        
        Such as o = o1*f1 + o2*f2 + ... + oj*fj + ... + oN*fN.
        
        To conclude, the frame of reference is fully described by o and the 
        basis transfer matrix noted 'mx', from the basis to the cartesian 
        basis.The transfer matrix size is N*N.
        
        mx = [e1 e2 ... ei ... eN] = [e11 e21 ... ei1 ... eN1
                                      e12 e22 ... ei2 ... eN2
                                      ... ... ... ... ... ...
                                      e1j e2j ... eij ... eNj
                                      ... ... ... ... ... ...
                                      e1N e2N ... eiN ... eNN]
        
        By this, a frame of reference is an sub-instance of Basis.
                                          
    Parameters
    -----------
        origin: np.ndarray with shape (N,) or list or tuple with len N
                The origin of the frame of reference expressed in the cartesian 
                frame of reference. 
    
        *E: np.ndarray with shape (N,) or list or tuple with len N 
            The ei basis vector expressed in the cartesian basis for i 
            in [1,N].      
            E = (e1,e2 ... eN)
            {N ei are requiered}
       
    Attributes
    ----------      
        origin: np.ndarray with shape (N,1)
                The origin of the frame of reference expressed in the cartesian 
                frame of reference. 
     
        Basis.Attributes:
                  
    Using
    ----------
        We want to create the frame of reference [o,e1,e2,e3] with: 
        o = [1        e1 = [1    e2 = [1    e3 = [0
             2              0          1          1
             1]             0]         0]         1]
            
        A solution is : 
            > e1 = np.array([1,0,0]) or [1,0,0] or (1,0,0)
            > e2 = np.array([1,1,0])
            > e3 = np.array([0,1,1])
            > o =  np.array([1,2,1])
            > frame = ReferenceFrame(o,e1,e2,e3)
            
        A other solution is :
            > E = [ [1,0,0] , [1,1,0] , [0,1,1] ]
            > o =  [1,2,1]
            > frame = ReferenceFrame(o,*E)
        
        Lets P a point whose coordinates (shape = (N,1)) are :
        P1 = [x1 in the cartesian frame of reference and P2 = [x2 in the frame. 
              y1                                               y2
              z1]                                              z2]
        
        We have the following relations : 
            > P1 = np.matmul(frame.mx, P2) + frame.origin
                 = frame.frame2cart(P2)
            > P2 = np.matmul(frame.imx, P1 - frame.origin) 
                 = frame.cart2frame(P1)
            
    Notes
    ----------
        Be carreful : eij = frame.mx[j-1,i-1]
    
        Do not modify coefficients of 'mx' or 'imx' directly.
        Use 'set_e', 'set_ei_coef' methods otherwise 'mx' and 'imx' won't be 
        in correspondance.
        
        You can also set 'mx' and 'imx' entirely. 
        
        If you had modify 'mx' or 'imx' coefficients with an other way, use 
        'update_mx' or 'update_imx' to correct the correspondance.              
    """
    def __init__(self, origin: Union[np.ndarray,list,tuple], 
                 *E: Union[np.ndarray,list,tuple]) -> None:
        super().__init__(*E)
        #origine of the frame of reference
        self.__origin = np.array(origin).astype(float).reshape((self.N,1))
        
    #======================================================================
    # --------- Setter and Getter methods  ---------
    #======================================================================    
    @property
    def origin(self) -> np.ndarray:
        """Returns the origin of the frame of refecence into an array of shape
        (N,1). """
        return self.__origin
    
    @origin.setter
    def origin(self, origin: Union[np.ndarray,list,tuple]) -> None:
        """
        Sets the origin of the frame of refecence with an new np.ndarray 
        with shape (N,1)
        
        Parameters
        -----------
            origin: np.ndarray with shape (N,) or list or tuple with len N
                    The origin of the frame of reference expressed in the 
                    cartesian frame of reference. 
        """
        self.__origin = np.copy(origin).reshape(self.__origin.shape)
        self.__origin = self.__origin.astype(float)
        
    #======================================================================
    # --------- overwritting methods ---------
    #======================================================================
    def __repr__(self) -> str:
        """ Returns the representation of the frame of reference."""
        s: str = f'{self.__class__.__name__} : N = {self.N}' 
        return s
    
    def __str__(self) -> str:
        """ Returns the string describing the frame of reference."""
        s: str = (f'{self.__class__.__name__} : N = {self.N}' 
                  + '\n-----------------')
        #first case : N <= 3 
        if self.N <= 3:
            s = s + '\n o = ['
            for i in range(0,self.N):
                s = s + f'{self.origin[i,0]:>8.2f} '
            s = s +']'
            for i in range(0,self.N):
                s = s + f'\ne{i+1} = ['
                for j in range(0,self.N):
                    s = s + f'{self.mx[j,i]:>8.2f} '
                s = s + ']'
        #second case : N > 3
        else: 
            s = s + (f'\n o = [{self.origin[0,0]:>8.2f} '
                  + f'{self.origin[1,0]:>8.2f} ... '
                  + f'{self.origin[self.N-1,0]:>8.2f} ]')
            s = s + (f'\ne{1} = [{self.mx[0,0]:>8.2f} '
                  + f'{self.mx[1,0]:>8.2f} ... '
                  + f'{self.mx[self.N-1,0]:>8.2f} ]'
                  + f'\ne{2} = [{self.mx[0,1]:>8.2f} '
                  + f'{self.mx[1,1]:>8.2f} ... '
                  + f'{self.mx[self.N-1,1]:>8.2f} ]'
                  + '\n  ... '
                  + f'\neN = [{self.mx[0,self.N-1]:>8.2f} '
                  + f'{self.mx[1,self.N-1]:>8.2f} ... '
                  + f'{self.mx[self.N-1,self.N-1]:>8.2f} ]')
        return s
    
    #======================================================================
    # --------- methods ---------
    #======================================================================
    def frame2cart(self, points: np.ndarray) -> np.ndarray:
        """ 
        Lets points an array of shape (N,M) whose correspond to M points 
        expressed in the frame. The function returns the cartesian coordinates 
        of the points in a array of shape (N,M).
        
        Lets P a point whose coordinates (shape = (N,1)) are :
        P1 = [x1 in the cartesian frame and P2 = [x2 in the frame. 
              y1                                  y2
              z1]                                 z2]
        
        We have the following relation : 
            > P1 = np.matmul(frame.mx, P2) + frame.origin 
                 = frame.frame2cart(X2)
        
        Parameters
        ----------
            points: np.ndarray with shape (N,M)
                     The M points expressed in the frame.
                 
        Return
        -----------
            points: np.ndarray with shape (N,M)
                     The M points expressed in the cartesian frame.        
        """
        return np.matmul(self.mx, points) + self.origin
    
    def cart2frame(self, points: np.ndarray) -> np.ndarray:
        """ 
        Lets points an array of shape (N,M) whose correspond to M points 
        expressed in the cartesian frame. The function returns the coordinates 
        of the points in the frame in a array of shape (N,M).
        
        Lets P a point whose coordinates (shape = (N,1)) are :
        P1 = [x1 in the cartesian frame and P2 = [x2 in the frame. 
              y1                                  y2
              z1]                                 z2]
        
        We have the following relation : 
            > P2 = np.matmul(frame.imx, P1 - frame.origin) 
                 = frame.cart2frame(P1)
        
        Parameters
        ----------
            points: np.ndarray with shape (N,M)
                     The M points expressed in the cartesian frame.
                 
        Return
        -----------
            points: np.ndarray with shape (N,M)
                     The M points expressed in the frame.        
        """
        return np.matmul(self.imx, points - self.origin) 
    
#=============================================================================
# --------- Commun Frame of Reference ---------
#=============================================================================
def cartesian_frame(N: int) -> ReferenceFrame :
    """
    Returns the cartesian frame of reference of R^N.
    
    Parameters
    ----------
        N: int
           The dimension of the vector space R^N.
    """
    l = []
    o = []
    for k in range(N):
        o.append(0)
        l.append([i == k for i in range(N)])
    return ReferenceFrame(o,*l)

def randn_frame(N: int) -> ReferenceFrame :
    """
    Returns a randn frame of reference of R^N.
    
    Parameters
    ----------
        N: int
           The dimension of the vector space R^N.
    """
    l = []
    o = []
    for k in range(N):
        o.append(random.random())
        l.append(np.random.randn(N))
    return ReferenceFrame(o,*l)

#==============================================================================
# --------- Test ---------
#==============================================================================
if __name__ == '__main__' : 
    frame = randn_frame(5)