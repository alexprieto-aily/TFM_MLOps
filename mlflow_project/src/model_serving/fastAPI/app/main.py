from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
import pandas as pd
import pickle

app = FastAPI()


class LoanData(BaseModel):
    annual_inc: float
    fico_range_high: float
    installment: float
    int_rate: float
    num_bc_sats: float
    num_bc_tl: float
    num_il_tl: float
    pub_rec: float
    pub_rec_bankruptcies: float
    revol_bal: float
    revol_util: float
    tot_coll_amt: float
    tot_cur_bal: float
    issue_d_month: float
    issue_d_year: float
    addr_state_AK: int
    addr_state_AL: int
    addr_state_AR: int
    addr_state_AZ: int
    addr_state_CA: int
    addr_state_CO: int
    addr_state_CT: int
    addr_state_DC: int
    addr_state_DE: int
    addr_state_FL: int
    addr_state_GA: int
    addr_state_HI: int
    addr_state_IA: int
    addr_state_ID: int
    addr_state_IL: int
    addr_state_IN: int
    addr_state_KS: int
    addr_state_KY: int
    addr_state_LA: int
    addr_state_MA: int
    addr_state_MD: int
    addr_state_ME: int
    addr_state_MI: int
    addr_state_MN: int
    addr_state_MO: int
    addr_state_MS: int
    addr_state_MT: int
    addr_state_NC: int
    addr_state_ND: int
    addr_state_NE: int
    addr_state_NH: int
    addr_state_NJ: int
    addr_state_NM: int
    addr_state_NV: int
    addr_state_NY: int
    addr_state_OH: int
    addr_state_OK: int
    addr_state_OR: int
    addr_state_PA: int
    addr_state_RI: int
    addr_state_SC: int
    addr_state_SD: int
    addr_state_TN: int
    addr_state_TX: int
    addr_state_UT: int
    addr_state_VA: int
    addr_state_VT: int
    addr_state_WA: int
    addr_state_WI: int
    addr_state_WV: int
    addr_state_WY: int
    application_type_Individual: int
    application_type_Joint_App: int
    home_ownership_ANY: int
    home_ownership_MORTGAGE: int
    home_ownership_NONE: int
    home_ownership_OTHER: int
    home_ownership_OWN: int
    home_ownership_RENT: int
    initial_list_status_f: int
    initial_list_status_w: int
    purpose_car: int
    purpose_credit_card: int
    purpose_debt_consolidation: int
    purpose_educational: int
    purpose_home_improvement: int
    purpose_house: int
    purpose_major_purchase: int
    purpose_medical: int
    purpose_moving: int
    purpose_other: int
    purpose_renewable_energy: int
    purpose_small_business: int
    purpose_vacation: int
    purpose_wedding: int
    sub_grade_A1: int
    sub_grade_A2: int
    sub_grade_A3: int
    sub_grade_A4: int
    sub_grade_A5: int
    sub_grade_B1: int
    sub_grade_B2: int
    sub_grade_B3: int
    sub_grade_B4: int
    sub_grade_B5: int
    sub_grade_C1: int
    sub_grade_C2: int
    sub_grade_C3: int
    sub_grade_C4: int
    sub_grade_C5: int
    sub_grade_D1: int
    sub_grade_D2: int
    sub_grade_D3: int
    sub_grade_D4: int
    sub_grade_D5: int
    sub_grade_E1: int
    sub_grade_E2: int
    sub_grade_E3: int
    sub_grade_E4: int
    sub_grade_E5: int
    sub_grade_F1: int
    sub_grade_F2: int
    sub_grade_F3: int
    sub_grade_F4: int
    sub_grade_F5: int
    sub_grade_G1: int
    sub_grade_G2: int
    sub_grade_G3: int
    sub_grade_G4: int
    sub_grade_G5: int
    term__36_months: int
    term__60_months: int
    verification_status_Not_Verified: int
    verification_status_Source_Verified: int
    verification_status_Verified: int
    disbursement_method_Cash: int
    disbursement_method_DirectPay: int



# Load your trained model
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@app.post("/predict_model1/")
async def predict(loan_data: List[LoanData]) -> Dict[str, List[int]]:
    try:
        pred_data = pd.DataFrame([loan.dict() for loan in loan_data])
        predictions = model.predict(pred_data)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    



if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

    
