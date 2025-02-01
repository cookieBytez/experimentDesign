import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import itertools

def perform_statistical_tests(statistical_significans_paths, model_names):
    """
    Performs McNemar's test on Hit Rate and ANOVA on other metrics
    
    Args:
        statistical_significans_paths: List of paths to CSV files containing 
            per-event metrics for different models
        model_names: List of names corresponding to the model results
    
    Returns:
        Dictionary containing test results
    """
    # Load all results into a dictionary of DataFrames
    results = {name: pd.read_csv(path) 
              for name, path in zip(model_names, statistical_significans_paths)}
    
    test_results = {'McNemar': {}, 'ANOVA': {}}
    
    # McNemar's test for Hit Rate (pairwise comparisons)
    for (model1, model2) in itertools.combinations(model_names, 2):
        # Extract hit values for both models
        hit1 = results[model1]['hit']
        hit2 = results[model2]['hit']
        
        # Create contingency table
        both_correct = np.sum((hit1 == 1) & (hit2 == 1))  # Both correct
        model1_only = np.sum((hit1 == 1) & (hit2 == 0))   # Model1 correct only
        model2_only = np.sum((hit1 == 0) & (hit2 == 1))   # Model2 correct only
        both_wrong = np.sum((hit1 == 0) & (hit2 == 0))    # Both wrong
        
        table = [[both_correct, model1_only],
                 [model2_only, both_wrong]]
        
        # Perform McNemar's test with continuity correction
        result = mcnemar(table, exact=False)
        test_results['McNemar'][f"{model1}_vs_{model2}"] = {
            'statistic': result.statistic,
            'pvalue': result.pvalue,
            'significant': result.pvalue < 0.05
        }
    
    # One-way ANOVA for other metrics
    for metric in ['precision', 'recall', 'RR', 'AP']:
        # Prepare data for ANOVA
        groups = [results[name][metric] for name in model_names]
        
        # Perform one-way ANOVA
        anova_result = f_oneway(*groups)
        anova_entry = {
            'statistic': anova_result.statistic,
            'pvalue': anova_result.pvalue,
            'significant': anova_result.pvalue < 0.05
        }
        
        # Add post-hoc Tukey HSD if ANOVA is significant
        if anova_entry['significant'] and len(model_names) > 2:
            combined = pd.concat([pd.Series(g, name=metric) 
                                for g in groups], axis=0)
            labels = np.repeat(model_names, [len(g) for g in groups])
            
            tukey = pairwise_tukeyhsd(combined.values, labels, 0.05)
            anova_entry['posthoc'] = str(tukey.summary())
        
        test_results['ANOVA'][metric] = anova_entry
    
    return test_results

def print_statistical_test_results(test_results):
    """
    Prints the results of McNemar's test and ANOVA in a nicely formatted way.
    
    Args:
        test_results: Dictionary containing statistical test results, 
                      as returned by perform_statistical_tests.
    """
    print("=== Statistical Test Results ===\n")
    
    # Print McNemar's test results
    print("McNemar's Test Results (Hit Rate):")
    if 'McNemar' in test_results and test_results['McNemar']:
        for comparison, result in test_results['McNemar'].items():
            print(f"  {comparison}:")
            print(f"    Statistic: {result['statistic']:.4f}")
            print(f"    P-value: {result['pvalue']:.4e}")
            print(f"    Significant: {'Yes' if result['significant'] else 'No'}")
    else:
        print("  No McNemar's test results available.")
    
    print("\nOne-Way ANOVA Results:")
    
    # Print ANOVA results
    if 'ANOVA' in test_results and test_results['ANOVA']:
        for metric, result in test_results['ANOVA'].items():
            print(f"  Metric: {metric}")
            print(f"    Statistic: {result['statistic']:.4f}")
            print(f"    P-value: {result['pvalue']:.4e}")
            print(f"    Significant: {'Yes' if result['significant'] else 'No'}")
            
            # If significant, show post-hoc Tukey HSD results
            if result['significant'] and 'posthoc' in result:
                print("    Post-hoc Tukey HSD Results:")
                print(result['posthoc'])
            elif result['significant']:
                print("    Post-hoc Tukey HSD Results: Not available.")
    else:
        print("  No ANOVA results available.")
    
    print("\n=== End of Results ===")



# Example usage:
results = perform_statistical_tests(
    ['statistical_significans_auto.csv',
     'statistical_significans_concat.csv',
     'statistical_significans_encode.csv',
     'statistical_significans_GRU4REC.csv',
     'statistical_significans_GRU4REC_concat.csv',
     'statistical_significans_popular.csv',
     'statistical_significans_random.csv'],
    ['auto', 'concat', 'encode',
     'GRU4REC', 'GRU4REC_concat',
     'popular', 'random']
)

print_statistical_test_results(results)