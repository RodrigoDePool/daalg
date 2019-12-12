#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import random

import argparse

import grafos05 as pd
  
###################################################################### main
def main(n_bits, len_sc, strng, c, l_coins, l_prob):
    """ args: num bits en sequence, num elements in sucesión SC
    """

    print("............................. cifrado merkle-hellman:")
    print("generando sucesiones ...")
    l_sc = pd.gen_super_crec(len_sc)
    min_print = 20
    print("\nfirst %d l_sc\n" % min_print, l_sc[ : min_print])
    
    p, q, m = pd.mod_mult_inv(l_sc)
    print("\np, q, mod\n", p, q, m)
    
    l_pub = pd.l_publica_2_l_super_crec(l_sc, p, m)
    print("\nfirst %d l_pub\n" % min_print, l_pub[ : min_print])
    
    _ = input("\npulsar Intro para continuar ....................\n")
    print("generando mensaje  ...")
    
    l_bits = pd.gen_random_bit_list(n_bits)
    print("\nfirst %d l_bits\n" % min_print, l_bits[ : min_print])
    
    _ = input("\npulsar Intro para continuar ....................\n")
    print("cifrando y descifrando  ...")
    
    l_cifra = pd.mh_encrypt(l_bits, l_pub, m)
    print("\nfirst %d l_cifra\n" % min_print, l_cifra[ : min_print])
    
    l_dec = pd.mh_decrypt(l_cifra, l_sc, q, m)
    print("\nfirst %d l_desc\n" % min_print, l_dec[ : min_print])
    
    len_l_pub = len(l_pub)
    print("\ndif_cifrado_descifrado", (np.array(l_bits[ : n_bits - len_l_pub]) - np.array(l_dec[ : n_bits - len_l_pub])).sum())
    
    _ = input("\npulsar Intro para continuar ....................\n")
    print("............................. longest_common_subsequence")
    print(pd.find_max_common_subsequence(strng, strng[ : : -1]))
    
    _ = input("\npulsar Intro para continuar ....................\n")
    print("............................. optimalchange")
    print("to change %d:\n" % c, pd.optimal_change(c, l_coins))
    
    _ = input("\npulsar Intro para continuar ....................\n")
    print("............................. search_tree")
    m_cost, m_order = pd.optimal_order(l_prob)
    print("roots_matrix:")
    print(m_order)
    
    print("\noptimal_insertion_order:")
    print(pd.list_opt_ordering_search_tree(m_order, 0, len(l_prob)-1))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cifrado y descifrado Merkle-Hellman.")
    
    parser.add_argument("-nb", "--num_bits", type=int, default=None, help="num bits a generar en lista a cifrar")
    parser.add_argument("-l", "--len_listas", type=int, default=100, help="longitud de listas publica y privada")
    parser.add_argument("-s", "--string", type=str, default="dabalearrozalazorraelabad", help="cadena para l_s_c")
    parser.add_argument("-c", "--change", type=int, default="53423", help="amount_to_change")
    parser.add_argument("-lc", "--l_coins", type=str, default="1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000", help="coins_to_give_change")
    parser.add_argument("-lp", "--l_prob", type=str, default="0.22,0.18,0.20,0.05,0.25,0.02,0.08", help="probs_of_keys_for_search_tree")
    
    args = parser.parse_args()
    
    l_coins = [int(c) for c in args.l_coins.split(',')]
    c = random.randint(0, sum(l_coins))
    
    #l_prob  = [float(c) for c in args.l_prob.split(',')]
    l_prob = list(np.array(l_coins)/sum(l_coins))
    
    if args.num_bits is not None:
        #main(args.num_bits, args.len_listas, args.string, args.change, l_coins, l_prob)
        main(args.num_bits, args.len_listas, args.string, c, l_coins, l_prob)
        
    else:
        parser.print_usage()
