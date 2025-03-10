from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger


class IssueTypes(Enum):
    # NOTE: We leave the accessability of "off-topic-samples"
    # via "irrelevants" to ensure backwards compatibility
    IRRELEVANTS = "irrelevants"
    OFF_TOPIC_SAMPLES = "off_topic_samples"
    NEAR_DUPLICATES = "near_duplicates"
    LABEL_ERRORS = "label_errors"


class IssueManager:
    def __init__(self, issue_dict: dict, meta_data_dict: Optional[dict] = None):
        self.issue_dict = issue_dict
        self.meta_data_dict = meta_data_dict if meta_data_dict is not None else {}

    def get_issues(
        self,
        issue_type: Union[str, IssueTypes],
        return_as_df: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame, None]:
        if issue_type is type(IssueTypes):
            issue_type = issue_type.value

        # NOTE: Backwards compatibility with "irrelevants"
        if issue_type == "irrelevants":
            issue_type = IssueTypes.OFF_TOPIC_SAMPLES.value

        sel_issues = self.issue_dict.get(issue_type, None)
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
            col_name_indices = [c for c in df.columns if "indices" in c]
            for k, v in self.meta_data_dict.items():
                if v is not None:
                    # for near duplicates there are multiple index columns
                    for c_index in col_name_indices:
                        if "_" in c_index:
                            df[f"{k}_{c_index}"] = df[c_index].apply(lambda x: v[x])
                        else:
                            df[k] = df[c_index].apply(lambda x: v[x])
            return df
        else:
            return sel_issues

    def __getitem__(self, key: str):
        return self.get_issues(issue_type=key, return_as_df=False)

    @property
    def keys(self):
        return self.issue_dict.keys()
