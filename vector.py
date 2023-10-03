""" @author: DAHERON FABIEN """
# TO DO: Typing verification
# TO DO: Test function

#=============================================================================
# --------- Import libraries ---------
#=============================================================================
import numpy as np
from typing import Union
from basis import Basis, cartesian_basis
import copy


#=============================================================================
# --------- Class Vector ---------
#=============================================================================
class Vector : 
    """
    Introducing
    ----------
        A vector of R^N is defined by its coordinates in a giving basis.
            
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
        
        The vector is noted v.
        We have v = [v1
                     v2
                     ...
                     vj coefficient number j
                     ...
                     vN]
        
        Such as v = v1*e1 + v2*e2 + ... + vj*ej + ... + vN*eN.
        
        However, the type 'Vector' allows to manipulate M vectors at the same 
        time. Fot it, the M vectors needs to be expressed in the same basis. 
        The M vectors are defined by their coordinates in a giving basis.
        The M vectors are noted vi for k in [1,M].
        Lets consider the matrix m with size N*M
        
        m = [v1 v2 ... vk ... vM] = [v11 v21 ... vk1 ... vM1
                                      v12 v22 ... vk2 ... vM2
                                      ... ... ... ... ... ...
                                      v1j v2j ... vkj ... vMj
                                      ... ... ... ... ... ...
                                      v1N v2N ... vkN ... vMN]
        
        Such as vk = vk1*e1 + vk2*e2 + ... + vkj*ej + ... + vkN*eN.
        
        We construct also mc the matrixe vector expressed in the cartesian 
        basis.
        
    Parameters
    -----------
        basis: Basis
               The basis {e1,e2, ..., eN} in witch the vectors are defined.
               
        *V: np.ndarray with shape (N,) or list or tuple with len N 
            The vi vector expressed in the given basis for i in [1,M].
            V = (v1,v2 ... vM)
            {M vi are requiered}
            
    Attributes
    ----------      
        basis: Basis
               The basis {e1,e2, ..., eN} in witch the vectors are defined
     
        m: np.ndarray with shape (N,M)
            The coordinates af the M vectors in the given basis.
       
        mc: np.ndarray with shape (N,M)
             The coordinates af the M vectors in the cartesian basis.
                               
        N: int
           The dimension of the vector space.
           
        M: int
           The number of vectors contained in the object.
        
    Using
    ----------
        We want to create 2 vectors v1 and v2 of dimension 3 in a given basis 
        named 'basis'.
        v1 = [1        v2 = [1
              2              0
              1]             0]
            
        A solution is : 
            > v1 = np.array([1,2,1]) or [1,2,1] or (1,2,1)
            > e2 = np.array([1,0,0])
            > vector = Vector(basis,v1,v2)
            
        A other solution is :
            > V = [ [1,2,1] , [1,2,1]]
            > vector = Vector(basis,*V)
        
        To change realize a change of basis, use :
            > vector.basis = new_basis
            
        If you have the matrix m or mc and you want to create a new Vector :
        You can use : 
            > new_vector = Vector(basis, np.zeros((basis.N,)))
            > new_vector.m = m
            
        
    Notes
    ----------
        Be carreful : vij = vector.m[j-1,i-1]
    
        Do not modify coefficients of 'm' or 'mc' directly.
        Use 'set_vector', 'set_vector_coef' methods otherwise 'm' and 'mc' 
        won't be in correspondance.
        
        You can also set 'm' and 'mc' entirely. 
        
        If you had modify 'm' or 'mc' coefficients 
        with an other way, use 'update_m' or 'update_mc' to correct the
        correspondance.              
    """
    def __init__(self, basis: Basis, *V: Union[np.ndarray,list,tuple]) -> None:
        self.__basis = copy.copy(basis)
        #coordinates of the vector
        for vi in V :
            vi = np.array(vi).astype(float).reshape((self.__basis.N,))
        self.__m = np.transpose(list(V)).astype(float)
        self.update_mc()
    
    #======================================================================
    # --------- Setter and Getter methods  ---------
    #======================================================================    
    @property
    def basis(self) -> Basis:
        """Returns the basis in witch the vectors are defined"""
        return self.__basis
    
    @basis.setter
    def basis(self, basis: Basis) -> None:
        """
        Changes the basis of the object. 
        Realizes the change of basis to expressed the matrix m in this new 
        basis.
        
        Parameters
        -----------
            basis: Basis
                   The basis {e1,e2, ..., eN} in witch the vectors are defined.
        """
        self.__basis = copy.copy(basis)
        self.__m = self.__basis.cart2basis(self.__mc)
        
    @property
    def N(self) -> int:
        """Returns the dimension of the R^N vector space."""
        return self.__m.shape[0]
        
    @property
    def M(self) -> int:
        """Returns the number of vector in the object"""
        return self.__m.shape[1]
    
    @property
    def m(self) -> np.ndarray:
        """Returns the matrix corresponding to the vectors coordinates in the 
        given basis"""
        return self.__m
    
    @m.setter
    def m(self, m: np.ndarray) -> None:
        """
        Sets the matrix corresponding to the vectors coordinates in the 
        given basis = self.basis.
        
        If you want to redefine m in a new basis, many solutions : 
            - change the basis and after set the matrix m
                > vector.basis = new_basis
                > vector.m = new_m {expressed in the new basis}
            - use the method 'set_m'
                > vector.set_m(new_basis, new_m)
                
        If you want to change only some coefficients of m please check methods
        'set_vector' and 'set_vector_coef' but do not change the matrix 
        directly! 
        
        If you need to change directly a coefficent, please use the method 
        update_mc after. {not recommended}
        
        See also add_vectors and del_vectors to add or del vectors in the 
        objet.
                
        Parameters
        -----------
            m: np.ndarray with shape (N,M)
                The coordinates af the M vectors in the given basis.
        """
        self.__m = np.copy(m).astype(float)
        self.update_mc()
        
    def set_m(self, basis: Basis, m: np.ndarray) -> None:
        """
        Sets the matrix corresponding to the vectors coordinates in an new 
        given basis.
                
        Parameters
        -----------
            basis: Basis
                   The basis {e1,e2, ..., eN} in witch the vectors are defined.
        
            m: np.ndarray with shape (N,M)
                The coordinates af the M vectors in the given basis.
        """
        self.__basis = copy.copy(basis)
        self.__m = np.copy(m).astype(float)
        self.update_mc()
    
    @property
    def mc(self) -> np.ndarray:
        """Returns the matrix corresponding to the vectors coordinates in the 
        cartesian basis"""
        return self.__mc
    
    @mc.setter
    def mc(self, mc: np.ndarray) -> None:
        """
        Sets the matrix corresponding to the vectors coordinates in the 
        cartesian basis.
        
        If you want to change only some coefficients of mc please check 
        methods 'set_vector' and 'set_vector_coef' but do not change the matrix 
        directly! 
        
        If you need to change directly a coefficent, please use the method 
        update_m after. {not recommended}
        
        See also add_vectors and del_vectors to add or del vectors in the
        objet.
        
        Parameters
        -----------
            mc: np.ndarray with shape (N,M)
                The coordinates af the M vectors in the cartesian basis.
        """
        self.__mc = np.copy(mc).astype(float)
        self.update_m()
    
    def set_vector(self, i: int, basis: Basis, vi: np.ndarray) -> None:
        """
        Sets the vector vi with an new np.ndarray with shape (N,).
        Coordinates of the vector vi is expressed in the given basis but will
        be converted into the basis self.basis.
        
        Parameters
        -----------
            i: int
               The vector vi to modify, i in [1,M].
               
            basis: Basis
                   The basis {e1,e2, ..., eN} in witch the vector is defined.
            
            vi: np.ndarray with shape (N,) or list or tuple with len N
                The vi vector expressed in the given basis.
        """
        #coordinates of the vector in the cartesian basis (cvi) and the
        #self.basis (vi)
        cvi = basis.basis2cart(np.array(vi).astype(float).reshape((self.N,))) 
        vi = self.__basis.cart2basis(cvi) #self.basis basis
        #Modification of m and mc
        self.__m[:,i-1] = vi
        self.__mc[:,i-1] = cvi
    
    def set_vector_coef(self, i: int, j: int,
                        basis: Basis, value: float) -> None:
        """
        Sets the coordinate j of the vector vi with an new float value.
        The value correspond of the vi.ej coordinate in the given basis 
        but will be converted into the basis self.basis.
        
        Parameters
        -----------
            i: int
               The vector vi to modify, i in [1,M].
               
            j: int
               The coefficient j of vi to modify, j in [1,N].
               
            basis: Basis
                   The basis {e1,e2, ..., eN} in witch the vector is defined.
            
            value: float
                   The value to set at the vij coefficient.
        """
        #vector vi in the given basis
        vi = basis.cart2basis(self.mc[:,i-1])
        vi[j-1] = float(value)
        #modification of v
        self.set_vector(i, basis, vi)
    
    #======================================================================
    # --------- overwritting methods ---------
    #======================================================================
    def __repr__(self) -> str:
        """ Returns the representation of the vector."""
        s: str = f'{self.__class__.__name__} : N = {self.N} ; M = {self.M}' 
        return s
    
    def __str__(self) -> str:
        """ Returns the string describing the vector."""
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
        self.__m = self.__basis.cart2basis(self.__mc)
        
    def update_mc(self) -> None:
        """Updates mc after a modification of m."""
        self.__mc = self.__basis.basis2cart(self.__m)
        
    def add_vectors(self, basis: Basis,
                    *V: Union[np.ndarray,list,tuple]) -> None:
        """
        Adds p new vectors in the object vector. 
        The vectors are expressed in the given basis but will be convert into 
        the basis self.basis.
        
        The new vectors are noted nV = (nv1,nv2, ..., nvp) such as the 
        new M attributs of is object become M + p. 
        
        The new vectors are placed after current vectors, such as the vectors
        V in the objet are :
        V = (v1,v2,...,vM,vM+1,...,vM+P) = (v1,v2,...,vM,nv1,nv2,...,nvp)
            
        Parameters
        -----------
            basis: Basis
                   The basis {e1,e2, ..., eN} in witch the vectors are defined.
                   
            *V: np.ndarray with shape (N,) or list or tuple with len N 
                The nvi new vector expressed in the given basis for i in [1,p].
                nV = (nv1,nv2, ..., nvp)
                {P vi are requiered}
        """
        #construction of the m and the mc matrix of new vector
        for vi in V :
            vi = np.array(vi).astype(float).reshape((self.__basis.N,))
        m_basis = np.transpose(list(V)).astype(float) #is the given basis
        mc = basis.basis2cart(m_basis) #in the cartesian basis
        m = self.__basis.cart2basis(mc) #in the basis of self
        #Concatenation of self.mc and mc, and concatenation of self.m and m
        self.__m = np.concatenate((self.__m,m),axis=1)
        self.__mc =  np.concatenate((self.__mc,mc),axis=1)
        
    def del_vectors(self, bool_array: Union[list,tuple,np.ndarray]) -> None:
        """
        Removes some vectors in the object.
        
        bool_array is an array of shape (M,) witch contains M booleans. 
        vectors corresponding to a False are removed.
            
        Parameters
        -----------
            bool_array: np.ndarray with shape (M,) or list or tuple with len M
                        The array or list containing M booleans (or 0 and 1). 
                        Only the vectors corresponding to False values are 
                        removed.
        """
        #construction of the m and the mc matrix of new vector
        bool_array = np.array(bool_array).astype(bool).reshape(self.M,)
        self.__m = self.__m[:,bool_array]
        self.__mc = self.__mc[:,bool_array]
    
    def norm(self) -> np.ndarray:
        """
        Returns an array with shape (M,) containing the norm of each vector of 
        the object.
        """
        return np.linalg.norm(self.mc,axis = 0)
    
    def normalize(self) -> None:
        """Normalizes each vector of the object"""
        #using mc and not __mc to use update_m
        self.mc = self.mc / self.norm()
        
    def scalar_product(self, vector: 'Vector') -> np.ndarray:
        """
        Returns the scalar product between the self object and vector. 
        
        if vector.M = 1:
            -> returns the the scalar product between each vector of self 
            and the vector.
            
            self : V = (v1,v2,...,vM)
            vector : U = (u1)
            
            return : V.U = (v1.u1,v2.u1,...,vM.u1)
        
        if vectors.M = self.M:
            -> returns the scalar product between corresponding vectors 
            of self and vector.
            
            self : V = (v1,v2,...,vM)
            vector : U = (u1,u2,...,uM)
            
            return : V.U = (v1.u1,v2.u2,...,vM.uM)
    
        Parameters
        ----------
            vector: Vector
                     The vector to scalarize with 'self'    
                 
        Returns
        ----------
            array: np.ndarray with shape (M,)
                   The scalar product between self and vector
        """
        assert self.N == vector.N , \
            "scalar product required vector.N = self.N"
        assert vector.M in [self.M,1] , \
            "scalar product required vector.M in [1,self.M] "
        return np.sum(self.mc*vector.mc,axis = 0)
    
    def cross_product(self, vector: 'Vector',
                      out: bool = True ) -> Union[None,'Vector']:
        """
        Computes the cross product between the self object and vector. 
        ONLY IF N == 3
        
        if vector.M = 1:
            -> computes the the cross product between each vector of self 
            and the vector.
            
            self : V = (v1,v2,...,vM)
            vector : U = (u1)
            
            return : V.U = (v1.u1,v2.u1,...,vM.u1)
        
        if vectors.M = self.M:
            -> computes the cross product between corresponding vectors 
            of self and vector.
            
            self : V = (v1,v2,...,vM)
            vector : U = (u1,u2,...,uM)
            
            return : V.U = (v1.u1,v2.u2,...,vM.uM)
        
        The basis of the return Vector is the cartesian basis.
    
        Parameters
        ----------
            vector: Vector
                     The vector to cross with 'self'    
            
            out: bool, optional 
                 The default is True.
                 If out = True : return the 'Vector' cross product and self 
                 is not changed. 
                 If out = False : return None and the result is put in self.
                 
                 
        Returns (if out = True)
        ----------
            V: Vector
                The cross product between self and vector 
                with basis = cartesian basis.
        """
        assert self.N == vector.N == 3 , \
            "cross product required N = 3"
        assert vector.M in [self.M,1] , \
            "cross product required vector.M in [1,self.M]"
        if out: 
            V = Vector(cartesian_basis(self.basis.N),
                       np.zeros((self.basis.N,)))
            V.mc = np.cross(self.mc,vector.mc,axis=0)
            return V
        else:
            self.mc = np.cross(self.mc,vector.mc,axis=0)
    
    def optical_reflection(self, normal: 'Vector',
                      out: bool = True ) -> Union[None,'Vector']:
        """
        Performes the optical reflection of the self object an an surface with 
        a given normal vector. 
        
        if normal.M = 1:
            -> Performes the optical reflection of each vector of self 
            on the vector.
            
            self : V = (v1,v2,...,vM)
            normal : N = (n1)
            
            return : V|N = (v1|n1,v2|n1,...,vM|n1)
        
        if normal.M = self.M:
            -> returns the cross product between corresponding vectors 
            of self and vector.
            
            self : V = (v1,v2,...,vM)
            normal : N = (n1,n2,...,nM)
            
            return : V|N = (v1|n1,v2|n2,...,vM|nM)
    
        Parameters
        ----------
            normal: Vector
                     The normal vectors on witch the reflection is calculate.
                     
            out: bool, optional 
                 The default is True.
                 If out = True : return the 'Vector' optical reflection and 
                 self is not changed.
                 If out = False : return None and the result is put in self.
        
        Returns (if out = True)
        ----------
            V: Vector
                The optical reflection of self on normal 
                with basis = cartesian basis.
        """
        # We note :
        #   - I the incident vector
        #   - N the normal vector
        #   - R the reflected vector
        #
        # We note . the scalar product
        # We have : R - I = k * N 
        # And : I.N = - R.N
        # 
        # + scalar product by N : (R-I).N = k* (N.N)
        #                       : R.N - I.N = k * |N|²
        #                       : k = -2*(I.N)/ |N|²
        #
        # Thus : R = I + k * N
        #      : R = I - [ 2 * (I.N) * N / |N|² ] 
        assert self.N == normal.N , \
            "scalar product required normal.N = self.N"
        assert normal.M in [self.M,1] , \
            "scalar product required normal.M in [1,self.M] "
        IdotN = self.scalar_product(normal)
        N2 = normal.norm()**2
        if out: 
            V = Vector(cartesian_basis(self.basis.N),
                       np.zeros((self.basis.N,)))
            
            V.mc = self.mc - (2 * IdotN * normal.mc / N2)
            return V
        else:
            self.mc = self.mc - (2 * IdotN * normal.mc / N2)
            
            
            
        
        
        
#=============================================================================
# --------- Commun Vectors ---------
#=============================================================================
def randn_vector(N: int, M: int, basis: Basis) -> Vector :
    """
    Returns M randn vector of R^N defined in the basis basis.
    
    Parameters
    ----------
        N: int
           The dimension of the vector space R^N.
           
        M: int 
           The number of vectors in the vector objet.
           
        basis : Basis
                The basis {e1,e2, ..., eN} in witch the vector is defined.
    """
    V = []
    for k in range(M):
        V.append(np.random.randn(N))
    return Vector(basis, *V)

#==============================================================================
# --------- Test ---------
#==============================================================================
if __name__ == '__main__' : 
    N = 3
    M = 10000
    vector = randn_vector(N, M, cartesian_basis(N))
    print('vector \n',vector,'\n')
    
    N = 3
    M = 10000
    vector2 = randn_vector(N, M, cartesian_basis(N))
    print('vector2 \n',vector2,'\n')
    
    print('v1.v2 \n',vector.scalar_product(vector2),'\n')
    
    print('v1 x v2 \n',vector.cross_product(vector2),'\n')
    
    print('vector (reflected) \n',vector.optical_reflection(vector2),'\n')

