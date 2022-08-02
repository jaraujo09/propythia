# -*- coding: utf-8 -*-
"""
##############################################################################

A class used for computing different types of DNA descriptors.
It contains descriptors from packages iLearn, iDNA4mC, rDNAse, ...

Authors: João Nuno Abreu

Date: 02/2022

Email:

##############################################################################
"""

import sys
from functools import reduce

class DNADescriptor:

    pairs = {
        'A': 'T',
        'T': 'A',
        'G': 'C',
        'C': 'G'
    }

    ALPHABET = 'ACGT'

    """
    The Descriptor class collects all descriptor calculation functions into a simple class.
    It returns the features in a dictionary object
    """

    def __init__(self, dna_sequence):
        """	Constructor """
        if(checker(dna_sequence)):
            self.dna_sequence = dna_sequence.strip().upper()
        else:
            seq = dna_sequence.strip().upper()
            self.dna_sequence = ''.join([letter for letter in seq if letter in self.ALPHABET])
            

    def get_length(self):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Calculates lenght of sequence (number of aa)
        :return: value of length
        """
        return len(self.dna_sequence)

    def get_gc_content(self):
        """
        Calculates gc content
        :return: value of gc content
        """
        gc_content = 0
        for letter in self.dna_sequence:
            if letter == 'G' or letter == 'C':
                gc_content += 1
        return round(gc_content / self.get_length(), 3)

    def get_at_content(self):
        """
        Calculates at content
        :return: value of at content
        """
        at_content = 0
        for letter in self.dna_sequence:
            if letter == 'A' or letter == 'T':
                at_content += 1
        return round(at_content / self.get_length(), 3)

    # ----------------------- NUCLEIC ACID COMPOSITION ----------------------- #

    def get_nucleic_acid_composition(self, normalize=True):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Calculates nucleic acid composition
        :param normalize: default value is False. If True, this method returns the frequencies of each nucleic acid.
        :return: dictionary with values of nucleic acid composition
        """
        res = make_kmer_dict(1)
        for letter in self.dna_sequence:
            res[letter] += 1

        if normalize:
            res = normalize_dict(res)
        return res

    def get_enhanced_nucleic_acid_composition(self, window_size=5, normalize=True):
        """
        From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6216033/#SM0, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Calculates enhanced nucleic acid composition
        :param normalize: default value is False. If True, this method returns the frequencies of each nucleic acid.
        :return: dictionary with values of enhanced nucleic acid composition
        """
        res = []
        for i in range(len(self.dna_sequence) - window_size + 1):
            aux_d = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
            segment = self.dna_sequence[i:i+window_size]

            for letter in segment:
                aux_d[letter] += 1

            if normalize:
                aux_d = normalize_dict(aux_d)

            res.append(aux_d)

        return res

    def get_dinucleotide_composition(self, normalize=True):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Calculates dinucleotide composition
        :param normalize: default value is False. If True, this method returns the frequencies of each dinucleotide.
        :return: dictionary with values of dinucleotide composition
        """
        res = make_kmer_dict(2)
        for i in range(len(self.dna_sequence) - 1):
            dinucleotide = self.dna_sequence[i:i+2]
            res[dinucleotide] += 1
        if normalize:
            res = normalize_dict(res)
        return res

    def get_trinucleotide_composition(self, normalize=True):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Calculates trinucleotide composition
        :param normalize: default value is False. If True, this method returns the frequencies of each trinucleotide.
        :return: dictionary with values of trinucleotide composition
        """
        res = make_kmer_dict(3)
        for i in range(len(self.dna_sequence) - 2):
            trinucleotide = self.dna_sequence[i:i+3]
            res[trinucleotide] += 1

        if normalize:
            res = normalize_dict(res)
        return res

    def get_k_spaced_nucleic_acid_pairs(self, k=0, normalize=True):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/
        Calculates k-spaced nucleic acid pairs
        :param k: value of k
        :param normalize: default value is False. If True, this method returns the frequencies of each k-spaced nucleic acid pair.
        :return: dictionary with values of k-spaced nucleic acid pairs
        """
        res = make_kmer_dict(2)
        for i in range(len(self.dna_sequence) - k - 1):
            k_spaced_nucleic_acid_pair = self.dna_sequence[i] + \
                self.dna_sequence[i+k+1]
            res[k_spaced_nucleic_acid_pair] += 1

        if normalize:
            res = normalize_dict(res)
        return res

    def get_kmer(self, k=2, normalize=True, reverse=False):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/, https://rdrr.io/cran/rDNAse/
        Calculates Kmer
        :param k: value of k
        :param normalize: default value is False. If True, this method returns the frequencies of all kmers.
        :param reverse: default value is False. If True, this method returns the reverse compliment kmer.
        :return: dictionary with values of kmer
        """
        res = make_kmer_dict(k)

        for i in range(len(self.dna_sequence) - k + 1):
            res[self.dna_sequence[i:i+k]] += 1

        if reverse:
            for kmer, _ in sorted(res.items(), key=lambda x: x[0]):
                reverse = "".join([self.pairs[i] for i in kmer[::-1]])

                # calculate alphabet order between kmer and reverse compliment
                if(kmer < reverse):
                    smaller = kmer
                    bigger = reverse
                else:
                    smaller = reverse
                    bigger = kmer

                # create in dict if they dont exist
                if(smaller not in res):
                    res[smaller] = 0
                if(bigger not in res):
                    res[bigger] = 0

                if(smaller != bigger):
                    # add to dict
                    res[smaller] += res[bigger]
                    # delete from dict
                    del res[bigger]

        if normalize:
            res = normalize_dict(res)

        return res

    def get_accumulated_nucleotide_frequency(self):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/, https://www.nature.com/articles/srep13859?proof=t%252Btarget%253D
        Calculates accumulated nucleotide frequency
        :return: list with values of accumulated nucleotide frequency
        """
        res = []
        aux_d = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        for i in range(len(self.dna_sequence)):
            aux_d[self.dna_sequence[i]] += 1
            x = round(aux_d[self.dna_sequence[i]] / (i + 1), 3)
            res.append(x)
        return res
    
    def get_accumulated_nucleotide_frequency_25_50_75(self, normalize=True):
        """
        From: https://pubmed.ncbi.nlm.nih.gov/31067315/, https://www.nature.com/articles/srep13859?proof=t%252Btarget%253D
        Calculates accumulated nucleotide frequency at 25%, 50% and 75%
        :return: list with values of accumulated nucleotide frequency
        """
        res = []
        d1 = make_kmer_dict(1)
        d2 = make_kmer_dict(1)
        d3 = make_kmer_dict(1)
        
        for letter in self.dna_sequence[:normal_round(len(self.dna_sequence) * 0.25)]:
            d1[letter] += 1
            
        for letter in self.dna_sequence[:normal_round(len(self.dna_sequence) * 0.50)]:
            d2[letter] += 1
        
        for letter in self.dna_sequence[:normal_round(len(self.dna_sequence) * 0.75)]:
            d3[letter] += 1
        res = [d1,d2,d3]

        if normalize:
            res = [normalize_dict(d1),normalize_dict(d2),normalize_dict(d3)]
        return res

    # --------------------------  Autocorrelation  -------------------------- #

    def get_DAC(self, phyche_index=["Twist", "Tilt"], nlag=2, all_property=False, extra_phyche_index=None):
        """Make DAC vector.

        CODE FROM repDNA (https://github.com/liufule12/repDNA)

        :param phyche_index: physicochemical properties list.
        :nlag: an integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA sequence in the dataset). It represents the distance between two dinucleotides.
        :param all_property: bool, choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string), and its corresponding value is a list. It means user-defined phyche_index.
        """
        k = 2
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)
        return make_ac_vector([self.dna_sequence], nlag, phyche_value, k)[0]

    def get_DCC(self, phyche_index=["Twist", "Tilt"], nlag=2, all_property=False, extra_phyche_index=None):
        """Make DCC vector.

        CODE FROM repDNA (https://github.com/liufule12/repDNA)

        :param phyche_index: physicochemical properties list.
        :nlag: an integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA sequence in the dataset). It represents the distance between two dinucleotides.
        :param all_property: bool, choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string), and its corresponding value is a list. It means user-defined phyche_index.
        """
        k = 2
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)
        return make_cc_vector([self.dna_sequence], nlag, phyche_value, k)[0]

    def get_DACC(self, phyche_index=["Twist", "Tilt"], nlag=2, all_property=False, extra_phyche_index=None):
        """Make DACC vector.

        CODE FROM repDNA (https://github.com/liufule12/repDNA)

        :param phyche_index: physicochemical properties list.
        :nlag: an integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA sequence in the dataset). It represents the distance between two dinucleotides.
        :param all_property: bool, choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string), and its corresponding value is a list. It means user-defined phyche_index.
        """
        k = 2
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)
        zipped = list(zip(make_ac_vector([self.dna_sequence], nlag, phyche_value, k),
                          make_cc_vector([self.dna_sequence], nlag, phyche_value, k)))
        vector = [reduce(lambda x, y: x + y, e) for e in zipped]

        return vector[0]

    def get_TAC(self, phyche_index=["Dnase I", "Nucleosome"], nlag=2, all_property=False, extra_phyche_index=None):
        """Make TAC vector.

        CODE FROM repDNA (https://github.com/liufule12/repDNA)

        :param phyche_index: physicochemical properties list.
        :nlag: an integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA sequence in the dataset). It represents the distance between two dinucleotides.
        :param all_property: bool, choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string), and its corresponding value is a list. It means user-defined phyche_index.
        """
        k = 3
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)
        return make_ac_vector([self.dna_sequence], nlag, phyche_value, k)[0]

    def get_TCC(self, phyche_index=["Dnase I", "Nucleosome"], nlag=2, all_property=False, extra_phyche_index=None):
        """Make TCC vector.

        CODE FROM repDNA (https://github.com/liufule12/repDNA)

        :param phyche_index: physicochemical properties list.
        :nlag: an integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA sequence in the dataset). It represents the distance between two dinucleotides.
        :param all_property: bool, choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string), and its corresponding value is a list. It means user-defined phyche_index.
        """
        k = 3
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)

        return make_cc_vector([self.dna_sequence], nlag, phyche_value, k)[0]

    def get_TACC(self, phyche_index=["Dnase I", "Nucleosome"], nlag=2, all_property=False, extra_phyche_index=None):
        """Make get_TACC vector.

        CODE FROM repDNA (https://github.com/liufule12/repDNA)

        :param phyche_index: physicochemical properties list.
        :nlag: an integer larger than or equal to 0 and less than or equal to L-2 (L means the length of the shortest DNA sequence in the dataset). It represents the distance between two dinucleotides.
        :param all_property: bool, choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string), and its corresponding value is a list. It means user-defined phyche_index.
        """
        k = 3
        phyche_value = ready_acc(k, phyche_index, all_property, extra_phyche_index)

        zipped = list(zip(make_ac_vector([self.dna_sequence], nlag, phyche_value, k),
                          make_cc_vector([self.dna_sequence], nlag, phyche_value, k)))
        vector = [reduce(lambda x, y: x + y, e) for e in zipped]

        return vector[0]

    # --------------------  PSEUDO NUCLEOTIDE COMPOSITION  -------------------- #

    def get_PseDNC(self, lamda=3, w=0.05):
        """
        From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Calculates the Pseudo Dinucleotide Composition Descriptor of DNA sequences
        :param lamda: value of lambda
        :param w: value of w
        :return: dictionary with values of Pseudo Dinucleotide Composition Descriptor
        """
        d = {
            'AA': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11],
            'AC': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
            'AG': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
            'AT': [1.07, 0.22, 0.62, -1.02, 2.51, 1.17],
            'CA': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
            'CC': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
            'CG': [-1.66, -1.22, -0.44, -0.82, -0.29, -1.39],
            'CT': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
            'GA': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
            'GC': [-0.08, 0.22, 1.33, -0.35, 0.65, 1.59],
            'GG': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
            'GT': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
            'TA': [-1.23, -2.37, -0.44, -2.24, -1.51, -1.39],
            'TC': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
            'TG': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
            'TT': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11]
        }

        counts = make_kmer_dict(2)
        for i in range(len(self.dna_sequence) - 1):
            dinucleotide = self.dna_sequence[i:i + 2]
            counts[dinucleotide] += 1
            
        fk = {k: v / sum(counts.values()) for k, v in counts.items()}
        all_possibilites = make_kmer_list(2)

        thetas = []
        L = len(self.dna_sequence)
        for i in range(lamda):
            big_somatorio = 0
            for j in range(L-i-2):
                somatorio = 0
                first_dinucleotide = self.dna_sequence[j:j+2]
                second_dinucleotide = self.dna_sequence[j+i+1:j+i+1+2]
                for k in range(6):
                    val = (d[first_dinucleotide][k] -
                           d[second_dinucleotide][k])**2
                    somatorio += val

                big_somatorio += somatorio/6

            # Theta calculation
            if(L-i-2 == 0):
                theta = 0
            else:
                theta = big_somatorio / (L-i-2)
            thetas.append(theta)

        # --------------------------------------------

        res = {}
        for i in all_possibilites:
            res[i] = round(fk[i] / (1 + w * sum(thetas)), 3)

        for i in range(lamda):
            res["lambda."+str(i+1)] = round(w * thetas[i] /
                                            (1 + w * sum(thetas)), 3)

        return res

    def get_PseKNC(self, k=3, lamda=1, w=0.5):
        """
        From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8138820/
        Calculates the Pseudo K Composition Descriptor of DNA sequences
        :param lamda: value of lambda
        :param w: value of w
        :return: dictionary with values of Pseudo K Composition Descriptor
        """
        d = {
            'AA': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11],
            'AC': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
            'AG': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
            'AT': [1.07, 0.22, 0.62, -1.02, 2.51, 1.17],
            'CA': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
            'CC': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
            'CG': [-1.66, -1.22, -0.44, -0.82, -0.29, -1.39],
            'CT': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
            'GA': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
            'GC': [-0.08, 0.22, 1.33, -0.35, 0.65, 1.59],
            'GG': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
            'GT': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
            'TA': [-1.23, -2.37, -0.44, -2.24, -1.51, -1.39],
            'TC': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
            'TG': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
            'TT': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11]
        }
        counts = make_kmer_dict(k)
        for i in range(len(self.dna_sequence) - k + 1):
            k_mer = self.dna_sequence[i:i + k]
            counts[k_mer] += 1
            
        fk = {k: v / sum(counts.values()) for k, v in counts.items()}
        all_possibilites = make_kmer_list(k)

        thetas = []
        L = len(self.dna_sequence)
        for i in range(lamda):
            big_somatorio = 0
            for j in range(L-i-2):
                somatorio = 0
                first_dinucleotide = self.dna_sequence[j:j+2]
                second_dinucleotide = self.dna_sequence[j+i+1:j+i+1+2]
                for k in range(6):
                    val = (d[first_dinucleotide][k] -
                           d[second_dinucleotide][k])**2
                    somatorio += val
                big_somatorio += somatorio/6

            # Theta calculation
            if(L-i-2 == 0):
                theta = 0
            else:
                theta = big_somatorio / (L-i-2)
            thetas.append(theta)

        # --------------------------------------------

        res = {}
        for i in all_possibilites:
            res[i] = round(fk[i] / (1 + w * sum(thetas)), 3)

        for i in range(lamda):
            res["lambda."+str(i+1)] = round(w * thetas[i] /
                                            (1 + w * sum(thetas)), 3)
        return res

    # ----------------------  CALCULATE ALL DESCRIPTORS  ---------------------- #

    def get_descriptors(self):
        """
        Calculates all descriptors
        :return: dictionary with values of all descriptors
        """
        res = {}
        res['length'] = self.get_length()
        res['gc_content'] = self.get_gc_content()
        res['at_content'] = self.get_at_content()
        res['nucleic_acid_composition'] = self.get_nucleic_acid_composition()
        # res['enhanced_nucleic_acid_composition'] = self.get_enhanced_nucleic_acid_composition()
        res['dinucleotide_composition'] = self.get_dinucleotide_composition()
        res['trinucleotide_composition'] = self.get_trinucleotide_composition()
        res['k_spaced_nucleic_acid_pairs'] = self.get_k_spaced_nucleic_acid_pairs()
        res['kmer'] = self.get_kmer()
        # res['accumulated_nucleotide_frequency'] = self.get_accumulated_nucleotide_frequency()
        res['accumulated_nucleotide_frequency'] = self.get_accumulated_nucleotide_frequency_25_50_75()
        res['DAC'] = self.get_DAC()
        res['DCC'] = self.get_DCC()
        res['DACC'] = self.get_DACC()
        res['TAC'] = self.get_TAC()
        res['TCC'] = self.get_TCC()
        res['TACC'] = self.get_TACC()
        res['PseDNC'] = self.get_PseDNC()
        res['PseKNC'] = self.get_PseKNC()
        return res

if __name__ == "__main__" or sys.path[0].split("/")[-1] == "descriptors":
    from utils import *
else:
    from .utils import *
