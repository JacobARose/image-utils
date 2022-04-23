"""

imutils/ml/utils/taxonomy_utils.py


Created on: Sunday April 10th, 2022  
Created by: Jacob Alexander Rose  

"""

import hydra
import pandas as pd
import pyarrow as pa
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import wandb
from typing import *

from imutils.ml.utils import template_utils


__all__ = ["TaxonomyLookupTable", "TaxonomyLoggerCallback"]



class TaxonomyLookupTable:
	"""
	Custom class designed as an interface for producing a table mapping unique taxonomy categories to their corresponding taxons above & below in the hierarchy.
	
	User passes in a dataframe containing their full data catalog (or at least a representative subset). The class takes 1 row from each unique value in the `smallest_taxon_col`, which conveniently also accomplishes the goal of having a minimum of 1 row per unique value in every other taxon above it.
	The default smallest_taxon_col is set to `Species`, which is defined as the concatenation of a `genus` and `species` column. The distinction between uppercase & lowercase `Species` vs. `species` is that 2 identical `species` names might refer to organisms in unrelated genera. The combination of `genus` with `species` is therefore guaranteed to prevent collisions in the namespace.
	
	Species->genus->family
	
	
	Features to be worked on:
	
	* Meerkat plugin for easily & dynamically visualizing images with taxonomy instance
	
	
	
	"""

	def __init__(self,
				 df: pd.DataFrame,
				 smallest_taxon_col: str="Species"):
		self.df = df
		self.smallest_taxon_col = smallest_taxon_col
		self.prepare_columns()

	def prepare_columns(self):
		df = self.df
		if (
			("Species" not in df.columns) and 
			("genus" in df.columns) and
			("species" in df.columns)
		):
			df = df.assign(Species = df.apply(lambda x: " ".join([x.genus, x.species]), axis=1))

		self.df = df.groupby(self.smallest_taxon_col).head(1)


	def query_rows_by_value(self,
							query_col: str,
							query_value: Any) -> pd.DataFrame:
		return self.df[self.df[query_col] == query_value]

	
	def lookup_table(self,
					 query_table: Union[pd.DataFrame, pd.Series],
					 on: str="Species") -> pd.DataFrame:
		"""
		Merge a query_table on the `on` column with this taxonomy table's dataframe, resulting in a new dataframe containing the query table & completed values from the lookup table.
		"""
		
		if isinstance(query_table, pd.Series):
			query_table = query_table.to_frame()
			
		lookup_table = self._setup_lookup_table(lookup_on=on)

		return query_table.merge(lookup_table, how='left', on=on)


	def _setup_lookup_table(self, lookup_on: str="Species") -> pd.DataFrame:
		"""
		Private method. User should call lookup_table() method instead.
		
		Removes sets of columns from taxonomy's underlying lookup table to prevent potentially invalid or irrelevant outputs.
		
		e.g. if looking up on genus or family values, any Species values will be nonsense, since there will usually be more than 1 valid Species.
		"""
		drop_cols = ["catalog_number", "path", "index", "collection"]
		if lookup_on in ["y", "family", "genus"]:
			drop_cols.extend(["species", "Species"])
		if lookup_on in ["y", "family"]:
			drop_cols.extend(["genus"])
		
		drop_cols = [c for c in drop_cols if c in self.df.columns]
		lookup_table = self.df.drop(columns=drop_cols)
		return lookup_table
	
	def state_dict(self):
		return {
			"df": self.df,
			"smallest_taxon_col": smallest_taxon_col
		}
	def to_pyarrow(self):
		return {
			"df": pa.Table.from_pandas(self.df),
			"smallest_taxon_col": str(self.smallest_taxon_col)
		}

	@classmethod
	def from_pyarrow(self, state_dict):
		state_dict = {
			"df": pa.Table.to_pandas(state_dict["df"]),
			"smallest_taxon_col": str(state_dict["smallest_taxon_col"])
		}
		return cls.load_from_state_dict(state_dict)
	
	@classmethod
	def load_from_state_dict(cls,
							 state_dict: Dict[str, Any]):
		return cls(**state_dict)
	

	def as_wandb_table(self) -> wandb.Table:
		return wandb.Table(dataframe=self.df)



class TaxonomyLoggerCallback(pl.Callback):
	
	def __init__(self,
				 name: str="taxonomy"):
		"""
		
		todo:
			1. Add a flag for optionally logging as an artifact instead.
		
		Arguments:
			name: str, default='taxonomy'
				name under which to log the table.
		
		"""
		self.name = name
		self.taxonomy_table = None
		super().__init__()
		
	def on_train_start(self, trainer, pl_module) -> None:
		
		df = trainer.datamodule.train_dataset.df
		
		self.taxonomy_table = TaxonomyLookupTable(df = df)
		
		logger = template_utils.get_wandb_logger(trainer.loggers)

		hydra.utils.log.info(f"Logging taxonomy table to run.")
		self.log_table(table=self.taxonomy_table.as_wandb_table(),
					   name=self.name,
					   global_step = trainer.global_step,
					   commit=True,
					   logger=logger)

	@staticmethod
	@rank_zero_only
	def log_table(table: wandb.Table,
				  name: str,
				  global_step: int=0,
				  commit: bool=True,
				  logger=None) -> None:

		if isinstance(logger, pl.loggers.WandbLogger):
			logger = logger.experiment
		else:
			logger = wandb

			wandb.log(
				{
					name: table,
					"global_step":global_step
				},
				commit=commit
			)