#=====================================================================================
"""
Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
#=====================================================================================

# utility library for sqlite relational database functionality
# use these functions to create meta, xyz, aev, and energy data tables
# and to write to or read from them

import time
import sys
import numpy as np
import os.path
import os
import io
import sqlite3
from sqlite3 import Error

verbose=False

class create_connection():
    """ Create a database connection to the SQLite database
        specified by db_file
        this is done in a class so that the connection closure is done automatically when
        one goes out of context.
        Similarly, allows adding other code as desired when closing, under __exit__()
    :param db_file: database file
    :return: Connection object
    """
    def __init__(self, db_file):
        self.db_file = db_file
    def __enter__(self):
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.db_file, detect_types=sqlite3.PARSE_DECLTYPES)
            self.conn.row_factory = sqlite3.Row
            return self.conn
        except Error as e:
            print(e)
            print("Cannot create a connection to database: ",database)
            sys.exit()
    def __exit__(self, type, value, traceback):
            self.conn.close()

class create_cursor():
    """ Create a cursor object to a given sqlite connection object
        this is done in a class so that the cursor closure is done automatically when
        one goes out of context.
        Similarly, allows adding other code as desired when closing, under __exit__()
    :param conn: connection object
    :return: cursor object
    """
    def __init__(self, conn):
        self.conn = conn
    def __enter__(self):
        try:
            self.cur = self.conn.cursor()
            return self.cur
        except Error as e:
            print(e)
            print("Cannot create a cursor object")
            sys.exit()
    def __exit__(self, type, value, traceback):
        self.cur.close()

def create_table(conn, table_name):
    """ Create a table with specified name
    :param conn: Connection object
    :param table_name: table name string
    :return:
    """

    if verbose:
        print('rdb: create_table:',table_name)

    # nb. for sqlite, default is to use date/time in GMT
    # Thus, for GMT         use:  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    # and,  for local time, use:  created_at TIMESTAMP DEFAULT (datetime('now','localtime'))
    dfltime = """created_at TIMESTAMP DEFAULT (datetime('now','localtime'))"""
    #dfltime = """created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"""

    if table_name == 'meta':
        # string specifying meta table structure
        sql = """ CREATE TABLE IF NOT EXISTS meta (
                        config TEXT NOT NULL,
                  """ + dfltime + """
                    );"""
    elif table_name == 'xyz':
        # string specifying xyz table structure
        sql = """ CREATE TABLE IF NOT EXISTS xyz (
                        id integer PRIMARY KEY,
                        calc text,
                        calc_params array,
                        temp real,
                        name text,
                        dist real,
                        geom array NOT NULL,
                  """ + dfltime + """
                    );"""
    elif table_name == 'aev':
        # sql string specifying aev table structure
        sql = """CREATE TABLE IF NOT EXISTS aev (
                        id integer PRIMARY KEY,
                        aevtyp array,
                        aev array NOT NULL,
                        daev array,
                        xyz_id integer NOT NULL,
                  """ + dfltime + """,
                        FOREIGN KEY (xyz_id) REFERENCES xyz (id)
                    );"""
    elif table_name == 'energy':
        # sql string specifying energy table structure
        sql = """CREATE TABLE IF NOT EXISTS energy (
                        id integer PRIMARY KEY,
                        fidelity integer,
                        sample_set_id integer,
                        calc text NOT NULL,
                        calc_params array,
                        E real NOT NULL,
                        force array,
                        Hessian array,
                        xyz_id integer NOT NULL,
                  """ + dfltime + """,
                        FOREIGN KEY (xyz_id) REFERENCES xyz (id)
                    );"""
    else:
        # table name not recognized, abort
        print("create_table: table_name",table_name,"not recognized")
        sys.exit()

    with create_cursor(conn) as cur:
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute(sql)
        conn.commit()

def adapt_array(arr):
    """ Convert a numpy array to a binary object (BLOB) for sqlite table storage
    : param arr: numpy array
    : return: sqlite blob
    """
    out = io.BytesIO()
    np.save(out, arr, allow_pickle=True)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    """ Convert an sqlite blob to a numpy array
    : param: sqlite blob
    : return: numpy array
    """
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle=True)

def insert_meta(conn, meta):
    """ Create a new row-instance in the meta table
    :param conn:
    :param meta:
    :return: meta id
    """
    if verbose:
        print('rdb: insert_meta:',meta)
    if meta[1] is None:
        meta = meta[0:1]
        sql = ''' INSERT INTO meta(config) VALUES(?) '''
    else:
        sql = ''' INSERT INTO meta(config,created_at) VALUES(?,?) '''

    with create_cursor(conn) as cur:
        cur.execute(sql, meta)
        conn.commit()
        return cur.lastrowid

def insert_xyz(conn, xyz):
    """ Create a new row-instance in the xyz table
    :param conn:
    :param xyz: tuple containing(calc,calc_params,temp,geom,datetime), where datetime can be None
    :      where calc is the name of the calculator (string)
    :      calc_params is a dictionary of the calculator kwargs
    :      temp is the temperature (float) which this xyz samples
    :      name is the unique name (string) of the species from which this sample was generated from
    :      dist is the distance from the base geometry = random number normalizing factor in NMS
    :      and where geom is a 2D numpy array containing xyz data
    :      xyz data for a CmHn molecule would have the first m rows being the (x,y,z) coords of the m C atoms
    :      and the last n rows being the (x,y,z) coords of the n H atoms
    :return: xyz id
    """
    if verbose:
        print('rdb: insert_xyz:')
        for i,s in enumerate(['calc','calc_params','temp','name','dist','geom','created_at']):
            print(s,":",xyz[i])

    nxyz = [x for x in xyz if x is not None]
    if verbose: print('rdb: nxyz:',nxyz)

    sxyz = ','.join([s for i,s in enumerate(['calc','calc_params','temp','name','dist','geom','created_at']) if xyz[i] is not None])
    if verbose: print('rdb: sxyz:',sxyz)

    sql  = ''' INSERT INTO xyz(''' + sxyz + ''') VALUES(''' + ','.join('?'*len(nxyz)) + ''') '''
    if verbose: print('rdb: sql :',sql)

    with create_cursor(conn) as cur:
        cur.execute(sql, nxyz)
        conn.commit()
        return cur.lastrowid

def insert_aev(conn, aev):
    """ Create a new row in the aev table
    :param conn:
    :param aev: tuple containing (aevtyp,aev,daev,xyz_id,created_at)
    :       where all except aev and xyz_id can be None
    :       aevtyp: aev metadata packaged as a numpy array
    :       aev: aev numpy array
    :       daev: derivative of aev w.r.t. x,y,z coords as a numpy array
    :       xyz_id: long integer being the id in the xyz table of this sample
    :       created_at: datetime
    :return:
    """
    if verbose:
        print('rdb: insert_aev:',aev)
    naev = [a for a in aev if a is not None]
    saev = ','.join([s for i,s in enumerate(['aevtyp','aev','daev','xyz_id','created_at']) if aev[i] is not None])
    sql  = ''' INSERT INTO aev(''' + saev + ''') VALUES(''' + ','.join('?'*len(naev)) + ''') '''

    with create_cursor(conn) as cur:
        cur.execute(sql, naev)
        conn.commit()
        return cur.lastrowid

def insert_energy(conn, energy):
    """ Create a new row in the energy table
    :param conn:
    :param energy: tuple containing (fidelity,sample_set_id,calc,calc_params,E,force,Hessian,xyz_id,created_at)
    :   where   fidelity is an integer labeling the level, the meaning is saved in meta
    :           sample_set_id is a mystery column for MLMF
    :           calc is the name of the quantum chemistry calculator (string)
    :           calc_params is a numpy array encapsulation of calculator parameters
    :           E is the energy, scalar
    :           force is the -grad(E) w.r.t. x,y,z coords
    :           Hessian is the second derivative mx of E w.r.t. the coords 
    :           xyz_id: long integer being the id in the xyz table of this sample
    :           created_at: datetime
    :return:
    """
    if verbose:
        print('rdb: insert_energy:',energy)
    nerg = [e for e in energy if e is not None]
    serg = ','.join([s for i,s in enumerate(['fidelity',
                                             'sample_set_id',
                                             'calc','calc_params',
                                             'E',
                                             'force',
                                             'Hessian',
                                             'xyz_id',
                                             'created_at']) if energy[i] is not None])
    sql  = ''' INSERT INTO energy(''' + serg + ''') VALUES(''' + ','.join('?'*len(nerg)) + ''') '''

    with create_cursor(conn) as cur:
        cur.execute(sql, nerg)
        conn.commit()
        return cur.lastrowid

def update_aev(conn, aev, key, val):
    """
    update aev row entries for all rows where key column entry matches given val
    :param conn:
    :param new/updated aev row tuple: 
    :param key string: e.g. 'calc', or 'xyz_id'
    :param val: value of the entry in the key column
    :return: last row id in table if changed something, else None
    """

    naev = [a for a in aev if a is not None]
    if not naev:
        return None    # no change being requested, return with None

    nae.append(val)

    saev = ' = ? ,\n\t\t'.join([s for i,s in enumerate(['aevtyp','aev','daev','xyz_id','created_at']) if aev[i] is not None])

    sql  = ''' UPDATE aev\n\tSET \t''' + saev + ''' = ?\n\tWHERE\t''' + key + ''' = ? '''

    with create_cursor(conn) as cur:
        cur.execute(sql, naev)
        conn.commit()
        return cur.lastrowid

def update_energy(conn, energy, key, val):
    """
    update energy row entries for all rows where key column entry matches given val
    a val of None implies make no change to this item
    :param conn:
    :param energy row tuple:  (fidelity,sample_set_id,calc,calc_params,E,force,xyz_id,created_at) where any of them can be None or the required new value
    : e.g. update_energy(conn,(None,None,'DFTC',None,None,None,None,None),'xyz_id',2) changes only the calc to 'DFTC' for rows whose xyz_id=2
    :param key string: e.g. 'calc', or 'xyz_id'
    :param val: value of the entry in the key column
    :return: last row id in table if changed something, else None
    """

    nerg = [e for e in energy if e is not None]
    if not nerg:
        return None    # no change being requested, return with None

    nerg.append(val)

    serg = ' = ? ,\n\t\t'.join([s for i,s in enumerate(['fidelity',
                                                        'sample_set_id',
                                                        'calc',
                                                        'calc_params',
                                                        'E',
                                                        'force',
                                                        'Hessian',
                                                        'xyz_id',
                                                        'created_at']) if energy[i] is not None])

    sql  = ''' UPDATE energy\n\tSET \t''' + serg + ''' = ?\n\tWHERE\t''' + key + ''' = ? '''

    with create_cursor(conn) as cur:
        cur.execute(sql, nerg)
        conn.commit()
        return cur.lastrowid

def update_xyz(conn, input, key=None, val=None):
    """
    update xyz row entries for all rows where key column entry matches given val
    a val of None implies make no change to this item
    :param conn:
    :param xyz row tuple:  (calc,calc_params,temp,name,dist,geom,created_at) where any of them can be None or the required new value
    : e.g. update_xyz(conn,(None,None,2000,None,None,None,None),'id',2) changes only the temp to 2000 for rows whose id=2
    :param key string: e.g. 'calc', or 'dist'
    :param val: value of the entry in the key column
    :return: last row id in table if changed something, else None
    """
    
    if input.__len__() > 7:
        if key is None and val is None:
            xyz = input[0:7]
            key = input[7]
            val = input[8]
        else:
            print('too many xyz row elements')
            sys.exit()
    else:
        xyz = input
    #print('xyz: {}\n\nkey: {}\n\nval: {}\n'.format(xyz, key, val))

    nxyz = [x for x in xyz if x is not None]
    if not nxyz:
        return None

    nxyz.append(val)

    sxyz = ','.join([s for i,s in enumerate(['calc','calc_params','temp','name','dist','geom','created_at']) if xyz[i] is not None])
    sxyz = ' = ? ,\n\t\t'.join([s for i,s in enumerate(['calc',
                                                        'calc_params',
                                                        'temp',
                                                        'name',
                                                        'dist',
                                                        'geom',
                                                        'created_at']) if xyz[i] is not None])

    sql  = ''' UPDATE xyz\n\tSET \t''' + sxyz + ''' = ?\n\tWHERE\t''' + key + ''' = ? '''

    with create_cursor(conn) as cur:
        cur.execute(sql, nxyz)
        conn.commit()
        return cur.lastrowid

def delete_data_by_key(conn, table, key, val):
    """
    Delete table data rows by key
    :param conn: Connection to the SQLite database
    :param table: table
    :param key: key
    :param val: value
    :return:
    """
    sql = 'DELETE FROM '+ table +' WHERE '+ key +'=?'
    with create_cursor(conn) as cur:
        cur.execute(sql, (val,))
        conn.commit()

def delete_data(conn,table):
    """
    Delete all rows in the specified table
    :param conn: Connection to the SQLite database
    :param table: string name of desired table
    :return:
    """
    sql = 'DELETE FROM ' + table
    with create_cursor(conn) as cur:
        cur.execute(sql)
        conn.commit()

def list_tables(conn):
    with create_cursor(conn) as cur:
        cur.execute('SELECT name from sqlite_master where type= "table"')
        return cur.fetchall()

def delete_table(conn,table):
    with create_cursor(conn) as cur:
        cur.execute('drop table if exists '+table)
        conn.commit()

def count_rows(conn,table):
    with create_cursor(conn) as cur:
        cur.execute('select COUNT(*) from '+table)
        return cur.fetchone()[0]

def select_data(conn,table):
    with create_cursor(conn) as cur:
        cur.execute('select * from '+table)
        return cur.fetchall()

def select_data_by_multi_key(conn, table, keyarr, valarr):
    sql ='select * from ' + table + ' where ' + ''.join([' AND '+key+'=?' if i > 0 else key+'=?' for i,key in enumerate(keyarr)])
    with create_cursor(conn) as cur:
        cur.execute(sql, tuple(valarr))
        return cur.fetchall()

def select_data_by_key_vals(conn, table, key, vals):
    sql=('select * from ' + table + ' where ' + key + ' in ({0})').format(', '.join('?' for _ in vals))
    with create_cursor(conn) as cur:
        cur.execute(sql, vals)
        return cur.fetchall()

def select_data_by_key(conn, table, key, val):
    sql='select * from ' + table + ' where ' + key + '=?'
    with create_cursor(conn) as cur:
        cur.execute(sql, (val,))
        return cur.fetchall()

def remove_duplicate_id(fname, fid):
    preamble()
    with create_connection(fname) as conn:
        crsr = conn.cursor()
        sql_query = f'DELETE FROM energy WHERE energy.id IN (SELECT MIN(energy.id) FROM energy WHERE energy.fidelity=? GROUP BY xyz_id HAVING COUNT(*) > 1)'
        crsr.execute(sql_query, fid)  # execute the filtering
        conn.commit()

def find_duplicate_id(fname, fid, verbose=True):
    xid = []
    preamble()
    with create_connection(fname) as conn:
        crsr = conn.cursor()
        sql_query = f'SELECT energy.xyz_id, MIN(energy.id) FROM energy WHERE energy.fidelity=? GROUP BY xyz_id HAVING COUNT(*) > 1;'
        record = crsr.execute(sql_query, fid)  # execute the filtering
        for r in record:
            xid.append([r['MIN(energy.id)'], r['xyz_id']])
    if verbose:
        print(fname, '\n', xid)

#def select_across_tables(conn, tables, keys, conditions):
#    """
#    :tables: list of tables to select from
#    :keys: list of keys to select
#            first key is in first table, second in second table, etc.
#    :conditions: criteria for filtering, can be any number
#    E.g.:
#    select_across_tables(conn, ['xyz', 'energy'], ['dist', 'E'], ['xyz.id=energy.xyz.id', 'energy.fidelity=1'])
#    this will provide an iterable for all entries where fidelity is 1 and correctly paired distances and energies
#    """
#    sql='SELECT' + ['{}.{}'.format(zl[0], zl[1]) for zl in list(zip(tables, keys))] + 'FROM' + xyz, energy 'WHERE' xyz.id = energy.xyz_id 'AND' energy.fidelity = {fidelity_val}';'
#    with create_curson(conn) as cur:
#        cur.execute(sql)
#        return cur.fetchall()


def preamble():
    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)

    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("array", convert_array)
	
#=================================================================================
# main routine

def main():

    #=============================================================================
    # create a database connection
    config   = 'test'
    database = config+'.db'
    preamble()

    # create a database connection
    with create_connection(database) as conn:
        # create meta table
        create_table(conn, 'meta')

        # create xyz table
        create_table(conn, 'xyz')

        # create aev table
        create_table(conn, 'aev')

        # create energy table
        create_table(conn, 'energy')

        print("Tables:",[table[:] for table in list_tables(conn)])

        # add metadata in meta table
        # configuration string
        insert_meta(conn, (config,None))

        # create new xyz entries
        xyz_id = []
        for i in range(10):
            calc = 'gaussian'
            data = {'opt': 'CalcFC, Tight',
                    'nprocshared': 8,
                    'chk': '651611561190890540062_well',
                    'basis': '6-31g',
                    'mem': '700MW',
                    'multiplicity': 2,
                    'scf': 'xqc',
                    'label':'C5H5/' + str(i),
                    'NoSymm': 'NoSymm',
                    'freq': 'freq',
                    'initial_magmoms': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                    'method': 'b3lyp'}
            calc_params   = np.array(list(data.items()))
            temp = 800.
            dist = 0.2
            name = 231323
            xyz_id.append( insert_xyz( conn, (calc,calc_params,temp,name,np.random.rand(),np.random.rand(2,3),None) ) )

        # create new aev entries for some chosen xyz ids
        aev_id = []
        aev_typ = np.random.randint(low=1,high=8,size=3)
        for id in xyz_id[1:4]:
            aev_id.append( insert_aev( conn, (aev_typ,np.random.rand(2,3),np.random.rand(2,3),id,None) ) )
        aev_typ = np.random.randint(low=1,high=8,size=3)
        for id in xyz_id[3:7]:
            aev_id.append( insert_aev( conn, (aev_typ,np.random.rand(2,3),np.random.rand(2,3),id,None) ) )

        # create new energy entries for some chosen xyz ids
        data = {'opt': 'CalcFC, Tight',
                'nprocshared': 8,
                'chk': '651611561190890540062_well',
                'basis': '6-31g',
                'mem': '700MW',
                'multiplicity': 2,
                'scf': 'xqc',
                'NoSymm': 'NoSymm',
                'freq': 'freq',
                'initial_magmoms': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                'method': 'b3lyp'}
        calc          = 'gaussian'
        calc_params   = np.array(list(data.items()))

        energy_id = []
        for id in xyz_id[1:4]:
            energy_id.append( insert_energy( conn, (1,2,calc,calc_params,np.random.rand(),np.random.rand(2,3),np.random.rand(2,2),id,None) ) )
        for id in xyz_id[2:7]:
            energy_id.append( insert_energy( conn, (2,2,calc,calc_params,np.random.rand(),np.random.rand(2,3),np.random.rand(2,2),id,None) ) )

        # example usage getting data from tables in different ways
        print("get all data from meta table:")
        for r in select_data(conn,'meta'):
            print(r[:])

        print("get all data from xyz table:")
        for r in select_data(conn,'xyz'):
            print(r[:])

        print("get all data from aev table:")
        for r in select_data(conn,'aev'):
            print(r[:])

        print("get all data from energy table:")
        for r in select_data(conn,'energy'):
            print(r[:])

        print("select energy by calc DFTA:")
        for r in select_data_by_key(conn,'energy','calc','nwchem'):
            print(r[:])

        print("select energy by xyz_id 3:")
        for r in select_data_by_key(conn,'energy','xyz_id',3):
            print(r[:])

        print("select energy by xyz_id 3 and calc gaussian:")
        for r in select_data_by_multi_key(conn,'energy',['xyz_id','calc'],[3,'gaussian']):
            print(r[:])

        print("update energy with id: 2 :")
        for r in select_data_by_key(conn,'energy','id',2):
            print(r[:])

        update_energy(conn,(3,4,'nwchem',np.array([2,3,5]),np.random.rand(),np.random.rand(2,3),np.random.rand(2,2),2,None),'id',2)
        for r in select_data_by_key(conn,'energy','id',2):
            print(r[:])

        print("update xyzid with xyz_id: 2 :")
        update_xyz(conn, (None, None, 2000, None, None, None, None, 'id', 2))
        print("xyz table:", ''.join("\n{0}".format(r[:]) for r in
                                       select_data_by_key(conn, 'xyz', 'id', 2)))

        print("update xyzid with xyz_id: 3 :")
        update_xyz(conn, (None, None, -2000, None, None, None, None), 'id', 3)
        print("xyz table:", ''.join("\n{0}".format(r[:]) for r in
                                       select_data_by_key(conn, 'xyz', 'id', 3)))

        print("update energy with xyz_id: 2 :")
        print('old:')
        for r in select_data_by_key(conn,'energy','xyz_id',2):
            print(r[:])

        update_energy(conn,(None,None,'DFTC',None,None,None,None,None,None),'xyz_id',2)

        for r in select_data_by_key(conn,'energy','xyz_id',2):
            print(r[:])

        print("energy table:",''.join("\n{0}".format(r[:]) for r in select_data_by_multi_key(conn,'energy',['xyz_id','calc'],[4,'gaussian'])))


        #print([r[:] for r in select_data(conn,'aev')])
        print("aev table:",''.join("\n{0}".format(r[:]) for r in select_data(conn,'aev')))
        delete_data_by_key(conn,'aev','xyz_id',2)
        print("aev table:",''.join("\n{0}".format(r[:]) for r in select_data(conn,'aev')))
        print("aev table xyz_id 4:",''.join("\n{0}".format(r[:]) for r in select_data_by_key(conn,'aev','xyz_id',4)))

        print("energy table:",''.join("\n{0}".format(r[:]) for r in select_data(conn,'energy')))
        delete_data_by_key(conn,'energy','xyz_id',2)
        print("energy table:",''.join("\n{0}".format(r[:]) for r in select_data(conn,'energy')))
        print("energy table xyz_id 4:",''.join("\n{0}".format(r[:]) for r in select_data_by_key(conn,'energy','xyz_id',4)))

    #=============================================================================

if __name__ == '__main__':
    main()

