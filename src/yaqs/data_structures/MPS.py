import numpy as np
import opt_einsum as oe

# Convention (sigma, chi_l-1, chi_l)
class MPS:
    def __init__(self, length: int, physical_dimensions: list=[], state: str='zeros'):
        self.tensors = []
        self.length = length
        self.physical_dimensions = physical_dimensions
        if not physical_dimensions:
            # Default case is the qubit (2-level) case
            for _ in range(length):
                self.physical_dimensions.append(2)
        assert len(physical_dimensions) == length

        # Create d-level |0> state
        for d in physical_dimensions:
            vector = np.zeros(d)
            if state == 'zeros':
                vector[0] = 1
            elif state == 'ones':
                vector[1] = 1
            else:
                raise ValueError("Invalid state string")

            tensor = np.expand_dims(vector, axis=(0, 1))

            tensor = np.transpose(tensor, (2, 0, 1))
            self.tensors.append(tensor)
        self.flipped = False
        self.orthogonality_center = 0

    def read_max_bond_dim(self) -> int:
        global_max = 0
        for tensor in self.tensors:
            local_max = max(tensor.shape[0], tensor.shape[2])
            if local_max > global_max:
                global_max = local_max

        return global_max

    def flip_network(self):
        """ Flips the bond dimensions in the network so that we can do operations
            from right to left

        Args:
            MPS: list of rank-3 tensors
        Returns:
            new_MPS: list of rank-3 tensors with bond dimensions reversed
                    and sites reversed compared to input MPS
        """
        new_tensors = []
        for tensor in self.tensors:
            new_tensor = np.transpose(tensor, (0, 2, 1))
            new_tensors.append(new_tensor)

        new_tensors.reverse()
        self.tensors = new_tensors
        self.flipped = not self.flipped

    def shift_orthogonality_center_right(self, current_orthogonality_center: int):
        """ Left and right normalizes an MPS around a selected site

        Args:
            MPS: list of rank-3 tensors with a physical dimension d^2
            selected_site: site of matrix M around which we normalize
        Returns:
            new_MPS: list of rank-3 tensors at each site
        """

        tensor = self.tensors[current_orthogonality_center]
        old_dims = tensor.shape
        matricized_tensor = np.reshape(tensor, (tensor.shape[0]*tensor.shape[1], tensor.shape[2]))
        Q, R = np.linalg.qr(matricized_tensor)
        Q = np.reshape(Q, (old_dims[0], old_dims[1], old_dims[2]))
        self.tensors[current_orthogonality_center] = Q

        self.orthogonality_center = current_orthogonality_center+1

        # If normalizing, we just throw away the R
        if current_orthogonality_center+1 < self.length:
            self.tensors[current_orthogonality_center+1] = oe.contract('ij, ajc->aic', R, self.tensors[current_orthogonality_center+1])

    def set_canonical_form(self, orthogonality_center: int):
        """ Left and right normalizes an MPS around a selected site
        Args:
            MPS: list of rank-3 tensors with a physical dimension d^2
            selected_site: site of matrix M around which we normalize
        Returns:
            new_MPS: list of rank-3 tensors at each site
        """
        def sweep_decomposition(orthogonality_center: int):
            for site, _ in enumerate(self.tensors):
                if site == orthogonality_center:
                    break
                self.shift_orthogonality_center_right(site)

        sweep_decomposition(orthogonality_center)
        self.flip_network()
        flipped_orthogonality_center = self.length-1-orthogonality_center
        sweep_decomposition(flipped_orthogonality_center)
        self.flip_network()
        self.orthogonality_center=orthogonality_center

    def normalize(self, form: str='A'):
        if form == 'B':
            self.flip_network()
        
        self.set_canonical_form(orthogonality_center=self.length-1)
        self.shift_orthogonality_center_right(self.length-1)

        if form == 'B':
            self.flip_network()

    def _check_canonical_form(self):
        """ Checks what canonical form an MPS is in if any

        Args:
            MPS: list of rank-3 tensors with a physical dimension d^2
        """
        A = np.conj(self.tensors)
        B = self.tensors

        A_truth = []
        B_truth = []
        epsilon = 1e-12
        for i in range(len(A)):
            M = oe.contract('ijk, ijl->kl', A[i], B[i])
            M[M < epsilon] = 0
            test_identity = np.eye(M.shape[0], dtype=complex)
            A_truth.append(np.allclose(M, test_identity))

        for i in range(len(A)):
            M = oe.contract('ijk, ibk->jb', B[i], A[i])
            M[M < epsilon] = 0
            test_identity = np.eye(M.shape[0], dtype=complex)
            B_truth.append(np.allclose(M, test_identity))

        if all(A_truth):
            print("MPS is left (A) canonical.")
            print("MPS is site canonical at site % d" % (len(MPS)-1))

        elif all(B_truth):
            print("MPS is right (B) canonical.")
            print("MPS is site canonical at site 0")

        else:
            sites = []
            for truth_value in A_truth:
                if truth_value:
                    sites.append(truth_value)
                else:
                    break
            for truth_value in B_truth[len(sites):]:
                sites.append(truth_value)

            try:
                print("MPS is site canonical at site % d." % sites.index(False))
            except:
                form = []
                for i in range(len(A_truth)):
                    if A_truth[i]:
                        form.append('A')
                    elif B_truth[i]:
                        form.append('B')
                print("The MPS has the form: ", form)
