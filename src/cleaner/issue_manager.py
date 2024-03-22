from enum import Enum
from typing import Optional, Union

import pandas as pd
from loguru import logger


class IssueTypes(Enum):
    IRRELEVANTS = "irrelevants"
    NEAR_DUPLICATES = "near_duplicates"
    LABEL_ERRORS = "label_errors"


class IssueManager:
    def __init__(self, issue_dict: dict, meta_data_dict: Optional[dict] = None):
        self.issue_dict = issue_dict
        for issue_type in IssueTypes:
            assert (
                issue_type.value in self.issue_dict
            ), f"{issue_type.value} not found in given dict."
        self.meta_data_dict = meta_data_dict if meta_data_dict is not None else {}

    def get_issues(
        self,
        issue_type: Union[str, IssueTypes],
        return_as_df: bool = False,
    ):
        if issue_type is type(IssueTypes):
            issue_type = issue_type.value

        sel_issues = self.issue_dict.get(issue_type)
        if sel_issues is None:
            return sel_issues

        if return_as_df:
            logger.warning("Returning as dataframe requires extensive memory.")
            df = pd.DataFrame()
            for k, v in sel_issues.items():
                if k == "auto_issues":
                    # `auto_issues` are given as list of indices to save memory
                    # thus need to be mapped back
                    df["auto_issues"] = False
                    df.loc[v, "auto_issues"] = True
                elif v.shape[-1] == 2:
                    for i in range(v.shape[-1]):
                        df[f"{k}_{i+1}"] = v[:, i]
                else:
                    df[k] = v
            for k, v in self.meta_data_dict.items():
                if v is not None:
                    # for near duplicates there are multiple index columns
                    for c_index in [c for c in df.columns if "indices" in c]:
                        if "_" in c_index:
                            df[f"{k}_{c_index}"] = df[c_index].apply(lambda x: v[x])
                        else:
                            df[k] = df[c_index].apply(lambda x: v[x])
            return df
        else:
            return self.issue_dict.get(issue_type)

    def __getitem__(self, key: str):
        return self.get_issues(issue_type=key, return_as_df=False)
