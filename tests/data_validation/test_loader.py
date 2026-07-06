import pandas as pd

from src.data.loader import load_complaints
from src.data.schema import REQUIRED_COLUMNS

_CSV_CONTENT = """odiNumber,manufacturer,crash,fire,numberOfInjuries,numberOfDeaths,dateOfIncident,dateComplaintFiled,vin,components,summary,products,make,model,modelYear
1,"Kia America, Inc.",True,False,0,0,03/24/2015,03/30/2015,5XYPG4A36GG,SEAT BELTS,Seat belt did not unlatch after crash.,[],KIA,SORENTO,2016
2,"Ford Motor Company",False,True,1,0,04/24/2015,04/27/2015,,ENGINE,Engine caught fire while idling.,[],FORD,F150,2017
"""


class TestLoadComplaints:
    def test_loads_all_required_columns_with_correct_dtypes(self, tmp_path):
        csv_path = tmp_path / "complaints.csv"
        csv_path.write_text(_CSV_CONTENT)

        df = load_complaints(path=csv_path)

        assert REQUIRED_COLUMNS <= set(df.columns)
        assert len(df) == 2
        assert df["crash"].dtype == "boolean"
        assert df["fire"].dtype == "boolean"
        assert pd.api.types.is_datetime64_any_dtype(df["dateOfIncident"])
        assert pd.api.types.is_datetime64_any_dtype(df["dateComplaintFiled"])

    def test_handles_missing_vin_as_null(self, tmp_path):
        csv_path = tmp_path / "complaints.csv"
        csv_path.write_text(_CSV_CONTENT)

        df = load_complaints(path=csv_path)

        assert df.loc[1, "vin"] is pd.NA or pd.isna(df.loc[1, "vin"])

    def test_boolean_values_parsed_correctly(self, tmp_path):
        csv_path = tmp_path / "complaints.csv"
        csv_path.write_text(_CSV_CONTENT)

        df = load_complaints(path=csv_path)

        assert df.loc[0, "crash"] == True  # noqa: E712
        assert df.loc[0, "fire"] == False  # noqa: E712
        assert df.loc[1, "fire"] == True  # noqa: E712
