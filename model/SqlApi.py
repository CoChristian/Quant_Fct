#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:29:29 2021

@author: Yitao Hu
"""
import datetime
import pandas as pd
import re
from sqlalchemy import create_engine
import pdb
from sqlalchemy import event
import numpy as np
import sqlalchemy


def add_own_encoders(conn, cursor, query, *args):
    cursor.connection.encoders[np.float64] = lambda value, encoders: float(value)


class DuplicatedDataEntryError(Exception):
    pass


class SQL_API():
    # default place to save duplicated data
    dup_data_engine = create_engine("mysql+pymysql://develop:haikuan_2025@192.168.110.5/dup_data")

    def __init__(self, read_engine=None, save_engine=None, dup_data_engine=dup_data_engine):
        """
        instantiate SQL API, pass in the read and save engine created from sqlalchemy

        Params:
        -------------------

        read_engine: sql engine to read data from

        save_engine: sql engine to save data

        dup_data_engine: sql engine to save duplicated data
        """
        # check engine arg types, if str create teh engine
        for engine_name, engine in zip(['read_engine', 'save_engine', 'dup_data_engine'],
                                       [read_engine, save_engine, dup_data_engine]):
            if isinstance(engine, str):
                setattr(self, engine_name, create_engine(engine))
            else:
                setattr(self, engine_name, engine)

        event.listen(self.save_engine, "before_cursor_execute", add_own_encoders)

    def read_data_from(self, query_info):

        """
        read data from sql database

        Params:

        -----------------------

        query_info: dict or str
            dict in the following format:

            {'method': 'select',
                 'sheet_name': 'daily_port_rtn_report2layer',
                 'tgt_field': {'way': 'max', 'field': ['beta_factor']},
                 'conditions': [{'field': 'trade_date',
                   'type': 'between',
                   'param': [20211201, 20211223]}]}

        specific params:

            'method': str, one of ["select", "delete"]

            'sheet_name': str, tablename in schema

            'tgt_field': dict in the following format:

                {'way': 'max', 'field': ['beta_factor']}

                specific params:

                    'way':str, method to aggregate data, one of ['show', "distinct", "count", "min", "max"]

                    'field': list of str, columns to select
            'conditions': list or tuple of dicts, each element is a condition,
                                list indicating AND logical operation
                                tuple indicating OR logical operation

                each element: dict of the following format:

                    "field": str, columns to filter on
                    "type": how to filer the field, one of ['big_equal', 'big', 'less',
                                                            'less_equal', 'equal', 'between', 'in', 'not_in']
                    "param": list of str, or list, the range to filter

            str as mysql query

        Returns:
        --------------

            table: pd.DataFrame

        """
        # translate to query
        if isinstance(query_info, dict):
            query = gen_sql_query(query_info)
        elif isinstance(query_info, str):
            query = query_info
        # print(query)
        table = pd.read_sql_query(query, con=self.read_engine)

        return table

    def file_exists(self, table_name):
        """
        Check whether a table exists in save schema or not

        """
        con = self.save_engine.connect()
        sql = "show tables;"
        con.execute(sql)
        engine = self.save_engine
        result = engine.execute(sql)
        tables = [result.fetchall()]
        table_list = re.findall('(\'.*?\')', str(tables))
        table_list = [re.sub("'", '', each) for each in table_list]

        if table_name in table_list:
            return True
        else:
            return False

    def save_data(self, df, tablename):
        """
        存数据，如果table 已经存在，覆盖掉
        :param df:
        :param tablename:
        :return:
        """
        if df.empty:
            print('input dataframe is zero size')
            return len(df)
        df.to_sql(name=tablename, con=self.save_engine, index=False, if_exists='replace')

    def insert_new_data_to(self, df, to_table):
        """
        Insert new data to a datatable in sql database

        Parameters
        ----------
        df : indexed pd.DataFrame
            pd.DataFrame with unique index for each row.
        to_table : str
            tablename of table to inseart data.
        engine: str of ['read' or 'save']
            str indicating which engine to insert data
        Returns
        -------
        None.

        """
        # check if data avaiable
        if df.empty:
            print("*" * 40)
            print('No new data to update for {to_table}'.format(to_table=to_table))
        # fetech index col names from df
        index_col = list(df.index.names)
        # get incremental index
        incremental_index = self.get_incremnetal_index(df, to_table, index_cols=index_col)
        # import pdb
        # pdb.set_trace()
        # reindex to original df to get incremental data
        try:
            incremental_df = df.reindex(incremental_index).copy().reset_index()
        except Exception as e:
            import pdb
            pdb.set_trace()
            pass
        if incremental_df.empty:
            print('*' * 40)
            print('No new data added to {to_table}'.format(to_table=to_table))
        else:
            # save the df to sql database
            incremental_df.to_sql(name=to_table, con=self.save_engine, index=False, if_exists='append')
            print('*' * 40)
            print('new data added to {to_table}, total {num_row} rows'.format(to_table=to_table,
                                                                              num_row=incremental_df.shape[0]))

    def get_unique_index(self, tablename, index_cols=['trade_date', 'ts_code'], engine='read'):
        """

        Get all the unique index or primary keys of a sql table

        tablename: str, sql table name

        index_cols: list of str, index columns

        Returns:
        ------------------------
            index_df: pd.DataFrame, a new datafame of all unique keys
        """
        # generate a select distinct statement
        query_info = {'method': 'select',
                      'sheet_name': tablename,
                      'tgt_field': {'way': 'distinct', 'field': index_cols},
                      'conditions': []}

        if len(index_cols) == 1:
            index_df = self.read_data_from(query_info)
        # if multiple index col get rid of '()'
        else:
            query = gen_sql_query(query_info).replace('(', ' ').replace(')', ' ')
            if engine == 'read':
                index_df = pd.read_sql_query(sql=query, con=self.read_engine)
            else:
                index_df = pd.read_sql_query(sql=query, con=self.save_engine)
        return index_df

    def get_incremnetal_index(self, from_table,
                              to_table,
                              index_cols=['trade_date', 'ts_code'],
                              from_engine='read',
                              to_engine='save',
                              df_format=False):
        """
        get the differnce index of two tables, generally used for updating data, by default
        from_table is in the read engine, while to_table is in the save engine

        Parameters
        ----------
        from_table : str or indexed pd.DataFrame
            table name to get the new search index .
        to_table : str
            table name to get the old search index.
        index_cols : list of str,
            index columns. The default is ['trade_date', 'ts_code'].
        engine : str, optional
            'read' or 'save'. The default is 'read'.

        Returns
        -------
        incremental_index : pd.Index or pd.MultiIndex
            incremental index of two tables.

        """

        if isinstance(from_table, str):
            # read in new index
            new_index = self.get_unique_index(from_table, index_cols=index_cols, engine=from_engine).set_index(
                index_cols).index
        elif isinstance(from_table, pd.DataFrame):
            new_index = from_table.index
        # readin old index
        # if old table exist, read old index
        if self.file_exists(to_table):
            old_index = self.get_unique_index(to_table, index_cols=index_cols, engine=to_engine).set_index(
                index_cols).index
            # get incremental index
            incremental_index = new_index.difference(old_index)
        # if old index not exist, use new index
        else:
            incremental_index = new_index.copy()
        if df_format:
            incremental_index = pd.DataFrame(index=incremental_index).reset_index()

        return incremental_index

    def check_duplicates(self, tablename, index_cols=['id']):
        """
        check whether the table has duplicated data entry

        Parameters
        ----------
        tablename : str
            tablename in the database.
        index_cols : list of str, optional
            index names of the table, must be unique for each data entry. The default is ['id'].


        """
        # generate statement
        query_stmt = """select {index_cols},{count_stmt}
                from {tablename}
                group by {index_cols}
                having
                    {condition_stmt}""".format(
            count_stmt=','.join(['count({col})'.format(col=col) for col in index_cols]),
            index_cols=','.join(index_cols),
            tablename=tablename,
            condition_stmt=' and '.join(['count({col})>1'.format(col=col) for col in index_cols]))
        check_df = pd.read_sql_query(query_stmt, con=self.save_engine)
        # if check df is empty, pass the test, else raise Error
        if check_df.empty:
            print('{tablename} has no duplicated entry'.format(tablename=tablename))
            return None
        else:
            print(check_df)
            raise DuplicatedDataEntryError('{tablename} has duplicated entries'.format(tablename=tablename))


def gen_sql_query(query_info):
    """
    Translate a Python dict to sql query to read data on some conditions

    Params:

    -----------------------

    query_info: dict in the following format:

        {'method': 'select',
             'sheet_name': 'daily_port_rtn_report2layer',
             'tgt_field': {'way': 'max', 'field': ['beta_factor']},
             'conditions': [{'field': 'trade_date',
               'type': 'between',
               'param': [20211201, 20211223]}]}

    specific params:

        'method': str, one of ["select", "delete"]

        'sheet_name': str, tablename in schema

        'tgt_field': dict in the following format:

            {'way': 'max', 'field': ['beta_factor']}

            specific params:

                'way':str, method to aggregate data, one of ['show', "distinct", "count", "min", "max"]

                'field': list of str, columns to select
        'conditions': list or tuple of dicts, each element is a condition,
                            list indicating AND logical operation
                            tuple indicating OR logical operation

            each element: dict of the following format:

                "field": str, columns to filter on
                "type": how to filer the field, one of ['big_equal', 'big', 'less',
                                                        'less_equal', 'equal', 'between', 'in', 'not_in']
                "param": list of str, or list, the range to filter

    Returns:
    --------------

        query: str, sql query format string

    """
    method = query_info['method']
    sheet_name = query_info['sheet_name']
    tgt_field = query_info["tgt_field"]
    field_way = tgt_field['way']
    fields = tgt_field['field']
    if len(fields):
        field_str = ",".join(fields)
    else:
        field_str = "*"
    assert method in ["select", "delete"]
    assert field_way in ['show', "distinct", "count", "min", "max"]
    if field_way == 'show':
        select_query = "%s %s from `%s`" % (method, field_str, sheet_name)
    else:
        select_query = "%s %s(%s) from `%s`" % (method, field_way, field_str, sheet_name)
    condition_query = translate_condition_info_2_query(query_info['conditions'])
    if condition_query != "":
        query = "%s where %s" % (select_query, condition_query)
    else:
        query = select_query
    return query


def translate_condition_info_2_query(condition_info):
    assert type(condition_info) in (tuple, list, dict)
    if type(condition_info) == tuple:
        logical = "or"
    elif type(condition_info) == list:
        logical = "and"
    else:
        logical = "element"
    if logical == "element":
        filed = condition_info['field']
        type_ = condition_info['type']
        param = condition_info['param']
        if type(param) == str:
            param = "'%s'" % param
        if type(param) == list:
            param = ["'%s'" % _ if type(_) == str else str(_) for _ in param]

        assert type_ in ['big_equal', 'big', 'less', 'less_equal', 'equal', 'between', 'in', 'not_in']
        if type_ == "big_equal":
            condition_str = "%s >= %s" % (filed, param)
        elif type_ == "big":
            condition_str = "%s >= %s" % (filed, param)
        elif type_ == "less":
            condition_str = "%s < %s" % (filed, param)
        elif type_ == "less_equal":
            condition_str = "%s <= %s" % (filed, param)
        elif type_ == "equal":
            condition_str = "%s = %s" % (filed, param)
        elif type_ == "between":
            condition_str = "%s between %s " % (filed, " and ".join(param))
        elif type_ == "in":
            condition_str = "%s in (%s) " % (filed, ",".join(param))
        elif type_ == "not_in":
            condition_str = "%s not in (%s) " % (filed, ",".join(param))
        else:
            pass
        condition_str = "(%s)" % condition_str
    elif logical == 'or':
        condition_list = []
        for _ in condition_info:
            condition_list.append(translate_condition_info_2_query(_))

        condition_str = " or ".join(condition_list)
        if len(condition_list) > 1:
            condition_str = "(%s)" % condition_str
    else:
        condition_list = []
        for _ in condition_info:
            condition_list.append(translate_condition_info_2_query(_))
        condition_str = " and ".join(condition_list)
        if len(condition_list) > 1:
            condition_str = "(%s)" % condition_str
    return condition_str
