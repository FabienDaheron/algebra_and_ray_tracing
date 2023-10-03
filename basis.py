""" @author: DAHERON FABIEN """
# TO DO: Typing verification
# TO DO: Test function

#=============================================================================
# --------- Import libraries ---------
#=============================================================================
import numpy as np
from typing import Union

#=============================================================================
# --------- Class Basis ---------
#=============================================================================
class Basis : 
    """
    Introducing
    ----------
        A basis of R^N is defined with N vectors of dimension N.
        The basis vectors are defined in the cartesian basis of R^N.
            
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
        
        To conclude, the basis is fully described by the transfer matrix, 
        noted 'mx', from the basis to the cartesian basis. The transfer matrix 
        size is N*N.
        
        mx = [e1 e2 ... ei ... eN] = [e11 e21 ... ei1 ... eN1
                                      e12 e22 ... ei2 ... eN2
                                      ... ... ... ... ... ...
                                      e1j e2j ... eij ... eNj
                                      ... ... ... ... ... ...
                                      e1N e2N ... eiN ... eNN]
                                          
    Parameters
    -----------
        *E: np.ndarray with shape (N,) or list or tuple with len N 
            The ei basis vector expressed in the cartesian basis for i 
            in [1,N].      
            E = (e1,e2 ... eN)
            {N ei are requiered}
       
    Attributes
    ----------           
        mx: np.ndarray with shape (N,N)
            The mx matrix corresponding to the transfer matrix from the basis 
            to the cartesian basis.   
             
        imx: np.ndarray with shape (N,N)
             The N*N matrix corresponding to the transfer matrix from the 
             cartesian basis to the basis.  
             imx = inv(mx)
             
        tmx: np.ndarray with shape (N,N)
             The transposition of mx.  
             tmx = transpose(mx)
             
        timx or itmx: np.ndarray with shape (N,N)
                      The transposition of imx.  
                      itmx or timx = transpose(inv(mx)) = inv(transpose(mx))
                      
        N: int
           The dimension N of the basis.
                  
    Using
    ----------
        We want to create the basis [e1,e2,e3] with: 
            e1 = [1    e2 = [1    e3 = [0
                  0          1          1
                  0]         0]         1]
            
        A solution is : 
            > e1 = np.array([1,0,0]) or [1,0,0] or (1,0,0)
            > e2 = np.array([1,1,0])
            > e3 = np.array([0,1,1])
            > basis = Basis(e1,e2,e3)
            
        A other solution is :
            > E = [ [1,0,0] , [1,1,0] , [0,1,1] ]
            > basis = Basis(*E)
        
        Lets V a vector whose coordinates (shape = (N,1)) are :
        X1 = [x1 in the cartesian basis and X2 = [x2 in the basis. 
              y1                                  y2
              z1]                                 z2]
        
        We have the following relation : 
            > X1 = np.matmul(basis.mx, X2) = basis.basis2cart(X2)
            > X2 = np.matmul(basis.imx, X1) = basis.cart2basis(X1)
            
    Notes
    ----------
        Be carreful : eij = basis.mx[j-1,i-1]    
    
        Do not modify coefficients of 'mx' or 'imx' directly.
        Use 'set_e', 'set_ei_coef' methods otherwise 'mx' and 'imx' won't be 
        in correspondance.
        
        You can also set 'mx' and 'imx' entirely. 
        
        If you had modify 'mx' or 'imx' coefficients with an other way, use 
        'update_mx' or 'update_imx' to correct the correspondance.
    """
    def __init__(self, *E: Union[np.ndarray,list,tuple]) -> None:
        #Transfer Matrix from the basis to the cartesian basis
        for ei in E : 
            ei = np.array(ei).astype(float).reshape((len(E),))
        self.__mx = np.transpose(list(E)).astype(float)
        #Transfer Matrix from the cartesian basis to the basis
        self.update_imx()
    
    #======================================================================
    # --------- Setter and Getter methods  ---------
    #======================================================================    
    @property
    def mx(self) -> np.ndarray:
        """Returns the transfer matrix from the basis to the cartesian 
        basis."""
        return self.__mx
    
    @mx.setter
    def mx(self, mx: np.ndarray) -> None:
        """
        Sets the matrix mx with an new np.ndarray with shape (N,N)
        
        Parameters
        -----------
            mx: np.ndarray with shape (N,N)
                The mx matrix corresponding to the transfer matrix from the 
                basis to the cartesian basis.
        """
        self.__mx = np.copy(mx).astype(float).reshape(self.__mx.shape)
        self.update_imx()
    
    @property
    def imx(self) -> np.ndarray:
        """Returns the transfer matrix from the cartesian basis to the 
        basis."""
        return self.__imx
    
    @imx.setter
    def imx(self, imx: np.ndarray) -> None:
        """
        Sets the matrix imx with an new np.ndarray with shape (N,N)
        
        Parameters
        -----------
            imx: np.ndarray with shape (N,N)
                The imx matrix corresponding to the transfer matrix from the 
                cartesian basis to the basis.
        """
        self.__imx = np.copy(imx).astype(float).reshape(self.__mx.shape)
        self.update_mx()
        
    @property
    def tmx(self) -> np.ndarray:
        """Returns the transposition of mx."""
        return np.transpose(self.__mx)
    
    @property
    def timx(self) -> np.ndarray:
        """Returns the transposition of imx."""
        return np.transpose(self.__imx)
    
    @property
    def N(self) -> int:
        """Returns the dimension of the R^N vector space."""
        return self.__mx.shape[0]
    
    def set_e(self, i: int, ei: Union[np.ndarray,list,tuple]) -> None:
        """
        Sets the basis vector ei with an new np.ndarray with shape (N,).
        
        Parameters
        -----------
            i: int
               The basis vector ei to modify, i in [1,N].
            
            ei: np.ndarray with shape (N,) or list or tuple with len N
                The ei basis vector expressed in the cartesian basis.
        """
        self.__mx[:,i-1] = np.array(ei).astype(float).reshape((self.N,))
        self.update_imx()
        
    def set_ei_coef(self, i: int, j: int, value: float) -> None:
        """
        Sets the coefficient j of the basis vector ei with an new value.
        
        Parameters
        -----------
            i: int
               The basis vector ei to modify, i in [1,N].
               
            j: int
               The coefficient j of ei to modify, j in [1,N].
              
            value: float
                The value to set at the eij coefficient. 
        """
        self.__mx[j-1,i-1] = float(value)
        self.update_imx()
        
    #======================================================================
    # --------- overwritting methods ---------
    #======================================================================
    def __repr__(self) -> str:
        """ Returns the representation of the basis."""
        s: str = f'{self.__class__.__name__} : N = {self.N}' 
        return s
    
    def __str__(self) -> str:
        """ Returns the string describing the basis."""
        s: str = (f'{self.__class__.__name__} : N = {self.N}' 
                  + '\n-----------------')
        #first case : N <= 3 
        if self.N <= 3:
            for i in range(0,self.N):
                s = s + f'\ne{i+1} = ['
                for j in range(0,self.N):
                    s = s + f'{self.mx[j,i]:>8.2f} '
                s = s + ']'
        #second case : N > 3
        else: 
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
    def update_imx(self) -> None:
        """Updates imx after a modification of mx."""
        self.__imx = np.linalg.inv(self.__mx)
        
    def update_mx(self) -> None:
        """Updates mx after a modification of imx."""
        self.__mx = np.linalg.inv(self.__imx)
    
    def e(self, i: int) -> np.ndarray:
        """
        Returns the basis vector ei.
        Don't use it function to modify mx -> use set_e or set_ei_coef !
        
        Parameters
        -----------
            i: int
               The basis vector ei to return, i in [1,N].
        
        Return
        -----------
            ei: np.ndarray with shape (N,)
                The ei basis vector expressed in the cartesian basis.
        """
        return self.mx[:,i-1]
    
    def normalize(self) -> None:
        """Normalizes the basis vectors."""
        self.mx = self.mx / np.linalg.norm(self.mx, axis = 0)
        
    def orthonormalize(self) -> None:
        """Orthonormalizes the basis vectors using Gram-Schmidt.
           The basis need to be free."""
        Q, R = np.linalg.qr(self.tmx)
        self.mx = np.transpose(Q)
        
    def is_free(self, epsilon: Union[int, float] = 1e-12) -> bool:
        """ 
        Returns True if the basis is free and False otherwise. 
        
        Parameters
        ----------
            epsilon: Union[int, float] such as epsilon > 0.0, optional 
                     The default is  1e-12.
                     The computing precision : A=B <=> |A - B| < epsilon.
        """
        # basis is free <-> det(mx) != 0 
        #                -> |det(mx)| > epsilon 
        return abs(np.linalg.det(self.mx)) > float(epsilon)
    
    def is_orthonormal(self, epsilon: Union[int, float] = 1e-12) -> bool:
        """ 
        Returns True if the basis is orthonormal and False otherwise. 
        
        Parameters
        ----------
            epsilon: Union[int, float] such as epsilon > 0.0, optional 
                     The default is  1e-12.
                     The computing precision : A=B <=> |A - B| < epsilon.
        """
        # basis is orthonormal <-> mx orthogonal
        #                      <-> mx * tmx = I3 
        #                      <-> tmx = imx
        #                       -> |tmx - imx| < epsilon
        #
        return np.linalg.norm(self.tmx - self.imx) < float(epsilon)
    
    def basis2cart(self, vectors: np.ndarray) -> np.ndarray:
        """ 
        Lets vectors an array of shape (N,M) whose correspond to M vectors 
        expressed in the basis. The function returns the cartesian coordinates 
        of the vectors in a array of shape (N,M).
        
        Lets V a vector whose coordinates (shape = (N,1)) are :
        X1 = [x1 in the cartesian basis and X2 = [x2 in the basis. 
              y1                                  y2
              z1]                                 z2]
        
        We have the following relation : 
            > X1 = np.matmul(basis.mx, X2) = basis.basis2cart(X2)
        
        Parameters
        ----------
            vectors: np.ndarray with shape (N,M)
                     The M vectors expressed in the basis.
                 
        Return
        -----------
            vectors: np.ndarray with shape (N,M)
                     The M vectors expressed in the cartesian basis.        
        """
        return np.matmul(self.mx, vectors)
    
    def cart2basis(self, vectors: np.ndarray) -> np.ndarray:
        """ 
        Lets vectors an array of shape (N,M) whose correspond to M vectors 
        expressed in the cartesian basis. The function returns the 
        coordinates of the vectors in the basis into a array of shape (N,M).
        
        Lets V a vector whose coordinates (shape = (N,1)) are :
        X1 = [x1 in the cartesian basis and X2 = [x2 in the basis. 
              y1                                  y2
              z1]                                 z2]
        
        We have the following relation : 
            > X2 = np.matmul(basis.imx, X1) = basis.cart2basis(X1)
        
        Parameters
        ----------
            vectors: np.ndarray with shape (N,M)
                     The M vectors expressed in the cartesian basis.
                 
        Return
        -----------
            vectors: np.ndarray with shape (N,M)
                     The M vectors expressed in the basis.        
        """
        return np.matmul(self.imx, vectors)
    
#=============================================================================
# --------- Commun Basis ---------
#=============================================================================
def cartesian_basis(N: int) -> Basis :
    """
    Returns the cartesian basis of R^N.
    
    Parameters
    ----------
        N: int
           The dimension of the vector space R^N.
    """
    l = []
    for k in range(N):
        l.append([i == k for i in range(N)])
    return Basis(*l)

def randn_basis(N: int) -> Basis :
    """
    Returns a randn basis of R^N.
    
    Parameters
    ----------
        N: int
           The dimension of the vector space R^N.
    """
    l = []
    for k in range(N):
        l.append(np.random.randn(N))
    return Basis(*l)

#==============================================================================
# --------- Test ---------
#==============================================================================
if __name__ == '__main__' : 
    basis = cartesian_basis(5)
    
    


