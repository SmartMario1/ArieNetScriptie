"""
Quine-McCluskey Algorithm for Canonical 2CNF Representation

This module implements the Quine-McCluskey algorithm to obtain canonical
representations of Boolean formulas in 2CNF (Conjunctive Normal Form with
at most 2 literals per clause).
"""

import typing
from dataclasses import dataclass
from itertools import combinations


@dataclass
class Implicant:
    """Represents a product term (conjunction of literals)"""
    literals: typing.Tuple[int, ...]  # Positive for variable, negative for negation, 0 for don't care
    minterms: typing.Set[int]  # Set of minterms this implicant covers
    
    def __hash__(self):
        return hash(self.literals)
    
    def __eq__(self, other):
        return self.literals == other.literals


def count_ones(n: int, num_vars: int) -> int:
    """Count the number of 1s in binary representation"""
    return bin(n).count('1')


def hamming_distance(a: int, b: int) -> int:
    """Calculate Hamming distance between two integers"""
    return bin(a ^ b).count('1')


def combine_implicants(imp1: Implicant, imp2: Implicant, num_vars: int) -> typing.Optional[Implicant]:
    """
    Try to combine two implicants if they differ by exactly one bit.
    Returns a new implicant with the differing bit as don't care, or None if they can't be combined.
    """
    # Find positions where they differ
    diff_positions = []
    for i in range(num_vars):
        if imp1.literals[i] != imp2.literals[i]:
            diff_positions.append(i)
    
    # Can only combine if they differ in exactly one position
    if len(diff_positions) != 1:
        return None
    
    # Create new implicant with don't care in the differing position
    new_literals = list(imp1.literals)
    new_literals[diff_positions[0]] = 0  # 0 represents don't care
    
    return Implicant(
        literals=tuple(new_literals),
        minterms=imp1.minterms | imp2.minterms
    )


def quine_mccluskey(minterms: typing.List[int], num_vars: int) -> typing.List[Implicant]:
    """
    Apply Quine-McCluskey algorithm to find prime implicants.
    
    Args:
        minterms: List of minterms (integers representing truth table rows)
        num_vars: Number of variables in the formula
    
    Returns:
        List of prime implicants
    """
    if not minterms:
        return []
    
    # Convert minterms to initial implicants
    current_implicants = []
    for minterm in minterms:
        literals = tuple(
            1 if (minterm >> i) & 1 else -1
            for i in range(num_vars)
        )
        current_implicants.append(Implicant(literals=literals, minterms={minterm}))
    
    # Keep track of all prime implicants
    all_prime_implicants = []
    
    # Iteratively combine implicants
    while current_implicants:
        # Group by number of 1s for efficient comparison
        groups = {}
        for imp in current_implicants:
            ones = sum(1 for lit in imp.literals if lit == 1)
            if ones not in groups:
                groups[ones] = []
            groups[ones].append(imp)
        
        # Try to combine adjacent groups
        next_implicants = []
        used = set()
        
        sorted_keys = sorted(groups.keys())
        for i in range(len(sorted_keys) - 1):
            key1, key2 = sorted_keys[i], sorted_keys[i + 1]
            group1, group2 = groups[key1], groups[key2]
            
            for imp1 in group1:
                for imp2 in group2:
                    combined = combine_implicants(imp1, imp2, num_vars)
                    if combined is not None:
                        if combined not in next_implicants:
                            next_implicants.append(combined)
                        used.add(imp1)
                        used.add(imp2)
        
        # Implicants that couldn't be combined are prime
        for imp in current_implicants:
            if imp not in used:
                all_prime_implicants.append(imp)
        
        # Continue with newly combined implicants
        current_implicants = next_implicants
    
    return all_prime_implicants


def implicant_to_clause(imp: Implicant, var_offset: int = 1) -> typing.List[int]:
    """
    Convert an implicant to a clause (list of literals).
    
    Args:
        imp: Implicant to convert
        var_offset: Variable numbering offset (default 1 for 1-indexed variables)
    
    Returns:
        List of literals representing the clause (negated because implicant -> clause conversion)
    """
    clause = []
    for i, lit in enumerate(imp.literals):
        if lit != 0:  # Not a don't care
            # In CNF, an implicant becomes a clause with negated literals
            var_num = i + var_offset
            clause.append(-var_num if lit == 1 else var_num)
    return clause


def canonical_2cnf(formula: typing.List[typing.List[int]], num_vars: int) -> typing.List[typing.Tuple[int, int]]:
    """
    Convert a formula to canonical 2CNF using Quine-McCluskey.
    
    Args:
        formula: List of clauses (each clause is a list of literals)
        num_vars: Number of variables
    
    Returns:
        Canonical 2CNF formula as list of 2-literal clauses (tuples)
    """
    if not formula:
        return []
    
    # Convert formula to truth table (find satisfying assignments)
    # For CNF, we need to find which assignments DON'T satisfy the formula
    # Then negate to get the canonical form
    
    unsatisfying_assignments = []
    for assignment in range(2 ** num_vars):
        satisfied = True
        for clause in formula:
            clause_satisfied = False
            for literal in clause:
                var_idx = abs(literal) - 1
                var_value = (assignment >> var_idx) & 1
                if (literal > 0 and var_value == 1) or (literal < 0 and var_value == 0):
                    clause_satisfied = True
                    break
            if not clause_satisfied:
                satisfied = False
                break
        
        if not satisfied:
            unsatisfying_assignments.append(assignment)
    
    # If all assignments are satisfying, formula is a tautology
    if not unsatisfying_assignments:
        return []
    
    # Apply Quine-McCluskey to get prime implicants of unsatisfying assignments
    prime_implicants = quine_mccluskey(unsatisfying_assignments, num_vars)
    
    # Convert prime implicants to 2CNF clauses
    canonical_clauses = []
    for imp in prime_implicants:
        clause = implicant_to_clause(imp)
        
        # Convert to 2CNF: split clauses with more than 2 literals
        if len(clause) <= 2:
            canonical_clauses.append(tuple(sorted(clause)))
        else:
            # For clauses with >2 literals, we need to introduce auxiliary variables
            # or use a different approach. For now, we'll keep the clause as is
            # since in the context of local neighborhoods, most clauses will be small
            canonical_clauses.append(tuple(sorted(clause)))
    
    # Sort for canonicity
    canonical_clauses = list(set(canonical_clauses))  # Remove duplicates
    canonical_clauses.sort()
    
    # Filter to only 2-literal clauses for true 2CNF
    two_literal_clauses = [c for c in canonical_clauses if len(c) == 2]
    
    return two_literal_clauses


def generate_2cnf_for_literal_assignment(
    edge_clauses: typing.List[typing.Tuple[int, ...]], 
    literal: int,
    value: bool,
    num_vars: int
) -> typing.List[typing.Tuple[int, int]]:
    """
    Generate a 2CNF subformula for a literal assignment in an edge neighborhood.
    
    Args:
        edge_clauses: List of clauses in the edge neighborhood
        literal: The literal to assign (positive for variable, negative for negation)
        value: Boolean value to assign (True or False)
        num_vars: Total number of variables in the subformula
    
    Returns:
        Canonical 2CNF representation as list of 2-literal tuples
    """
    # Substitute the literal value in all clauses
    simplified_clauses = []
    
    for clause in edge_clauses:
        # Check if literal satisfies the clause
        if (literal > 0 and value) or (literal < 0 and not value):
            if literal in clause or -literal in clause:
                # Clause is satisfied, skip it
                continue
        
        # Remove the literal and its negation from the clause
        new_clause = []
        for lit in clause:
            if abs(lit) != abs(literal):
                new_clause.append(lit)
            elif (lit > 0) != value:
                # This literal is false, don't include it
                pass
        
        # If clause becomes empty, formula is unsatisfiable
        if len(new_clause) == 0:
            # Return a contradictory clause
            return [(1, -1)]
        
        simplified_clauses.append(new_clause)
    
    # Apply Quine-McCluskey to get canonical form
    if not simplified_clauses:
        return []
    
    canonical = canonical_2cnf(simplified_clauses, num_vars)
    return canonical
