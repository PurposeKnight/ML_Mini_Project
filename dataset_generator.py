"""
Dataset Generator for Freddie Mac Loan-Level Data & Macroeconomic Factors.
Generates synthetic data mirroring the schema and distributions in the Stanford paper.
"""

import numpy as np
import pandas as pd
from config import RANDOM_SEED, NUM_LOANS, MONTHS_PER_LOAN, PREPAYMENT_RATIO, DATA_PATH


def generate_mbs_dataset(num_loans=NUM_LOANS, avg_months=MONTHS_PER_LOAN, seed=RANDOM_SEED):
    """
    Generates realistic loan-level origination, monthly performance, and macro data.
    """
    np.random.seed(seed)
    records = []

    occupancy_types = ['P', 'I', 'S']
    occupancy_probs = [0.85, 0.10, 0.05]
    
    purpose_types = ['P', 'C', 'N'] # Purchase, Cash-out refi, No-cash refi
    purpose_probs = [0.55, 0.25, 0.20]

    channel_types = ['R', 'B', 'C'] # Retail, Broker, Correspondent
    channel_probs = [0.50, 0.20, 0.30]

    for loan_id in range(100000, 100000 + num_loans):
        credit_score = int(np.clip(np.random.normal(735, 55), 450, 850))
        orig_cltv = float(np.clip(np.random.normal(76, 12), 30, 105))
        dti = float(np.clip(np.random.normal(36, 9), 10, 65))
        orig_interest_rate = round(float(np.clip(np.random.normal(4.25, 0.75), 2.5, 7.5)), 3)
        orig_upb = float(np.random.choice(np.arange(100000, 650000, 10000)))
        
        occupancy = np.random.choice(occupancy_types, p=occupancy_probs)
        purpose = np.random.choice(purpose_types, p=purpose_probs)
        channel = np.random.choice(channel_types, p=channel_probs)
        first_time = np.random.choice(['Y', 'N'], p=[0.35, 0.65])
        
        # Monthly progression
        num_months = np.random.randint(3, avg_months * 2)
        curr_upb = orig_upb

        prepaid_flag = False
        for month in range(1, num_months + 1):
            if prepaid_flag:
                break
            
            loan_age = month * 3 # 3-month steps
            months_to_maturity = max(0, 360 - loan_age)
            
            # Amortization reduction
            curr_upb = max(0.0, curr_upb * (1 - 0.003 * (1 + loan_age / 360.0)))
            
            # Macro environment
            mtgrate = round(float(np.clip(orig_interest_rate + np.random.normal(-0.5, 0.8), 2.25, 7.0)), 3)
            unemployment_rate = round(float(np.clip(np.random.normal(5.0, 1.2), 3.0, 10.0)), 2)
            hpi_appreciation = round(float(np.clip(np.random.normal(4.5, 2.0), -5.0, 12.0)), 2)
            
            rate_diff = orig_interest_rate - mtgrate # positive means refinancing is attractive!
            
            # True statistical model for prepayment (logit)
            logit = (
                -4.5
                + 1.8 * rate_diff
                + 0.004 * (credit_score - 700)
                - 0.015 * (orig_cltv - 75)
                + 0.02 * (loan_age / 12.0)
                + 0.000002 * (curr_upb - 200000)
                + (0.4 if occupancy == 'P' else -0.2)
                + (0.3 if purpose in ['C', 'N'] else 0.0)
            )
            prob_prepay = 1.0 / (1.0 + np.exp(-logit))
            
            # Scale probability to hit target overall prepayment ratio
            prob_prepay = np.clip(prob_prepay * 0.35, 0.001, 0.95)
            
            is_prepaid = 1 if np.random.rand() < prob_prepay else 0
            if is_prepaid == 1:
                prepaid_flag = True

            records.append({
                'loan_id': loan_id,
                'credit_score': credit_score,
                'orig_CLTV': orig_cltv,
                'dti': dti,
                'orig_interest_rate': orig_interest_rate,
                'orig_upb': orig_upb,
                'occupancy_status': occupancy,
                'loan_purpose': purpose,
                'channel': channel,
                'first_time_homebuyer': first_time,
                'current_UPB': curr_upb,
                'loan_age': loan_age,
                'months_to_maturity': months_to_maturity,
                'current_interest_rate': orig_interest_rate,
                'mtgrate': mtgrate,
                'unemployment_rate': unemployment_rate,
                'hpi_appreciation': hpi_appreciation,
                'rate_diff': rate_diff,
                'is_prepaid': is_prepaid
            })

    df = pd.DataFrame(records)
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated dataset with {len(df)} records across {df['loan_id'].nunique()} loans.")
    print(f"Prepayment count: {df['is_prepaid'].sum()} ({df['is_prepaid'].mean()*100:.2f}% of total).")
    return df


if __name__ == "__main__":
    generate_mbs_dataset()
