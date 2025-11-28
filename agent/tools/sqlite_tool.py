"""
SQLite Tool for Northwind Database
Provides schema introspection and safe SQL execution
"""
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional


class SQLiteTool:
    """Interface to the Northwind SQLite database"""
    
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row  # Access columns by name
        print(f"Connected to database: {self.db_path}")
    
    def get_schema(self) -> str:
        """
        Get database schema as formatted text for LLM prompts
        
        Returns:
            String describing all tables and their columns
        """
        cursor = self.connection.cursor()
        
        # Get all tables (including views)
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type IN ('table', 'view') 
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_text = "Database Schema:\n\n"
        
        for table in tables:
            # Get column info for each table
            cursor.execute(f'PRAGMA table_info("{table}")')
            columns = cursor.fetchall()
            
            schema_text += f"Table: {table}\n"
            schema_text += "Columns:\n"
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                schema_text += f"  - {col_name} ({col_type})\n"
            schema_text += "\n"
        
        return schema_text
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type IN ('table', 'view')
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        return [row[0] for row in cursor.fetchall()]
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """
        Execute a SQL query safely and return results
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Dict with keys:
                - success (bool): Whether query executed without error
                - columns (list): Column names
                - rows (list): List of dicts, one per row
                - error (str): Error message if failed
        """
        # Basic safety checks
        sql_upper = sql.upper().strip()
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return {
                    "success": False,
                    "columns": [],
                    "rows": [],
                    "error": f"Query contains forbidden keyword: {keyword}"
                }
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            
            # Get column names
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                
                # Fetch all rows as dicts
                rows = []
                for row in cursor.fetchall():
                    row_dict = {}
                    for idx, col in enumerate(columns):
                        row_dict[col] = row[idx]
                    rows.append(row_dict)
                
                return {
                    "success": True,
                    "columns": columns,
                    "rows": rows,
                    "error": ""
                }
            else:
                return {
                    "success": True,
                    "columns": [],
                    "rows": [],
                    "error": ""
                }
        
        except Exception as e:
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "error": str(e)
            }
    
    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """
        Validate SQL without executing it (using EXPLAIN)
        
        Returns:
            Dict with keys: valid (bool), error (str)
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            return {"valid": True, "error": ""}
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("Database connection closed")


# Singleton instance
_db_instance = None


def get_db(db_path: str = "data/northwind.sqlite") -> SQLiteTool:
    """Get or create the global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = SQLiteTool(db_path)
    return _db_instance


if __name__ == "__main__":
    # Test the SQL tool
    print("Testing SQLite Tool...\n")
    
    db = get_db()
    
    # Test 1: Get schema
    print("=== Schema ===")
    schema = db.get_schema()
    print(schema[:500] + "...\n")  # Print first 500 chars
    
    # Test 2: Get table names
    print("=== Tables ===")
    tables = db.get_table_names()
    print(f"Found {len(tables)} tables: {', '.join(tables)}\n")
    
    # Test 3: Execute simple query
    print("=== Test Query: Top 5 Products ===")
    result = db.execute_query("SELECT ProductName, UnitPrice FROM Products LIMIT 5")
    
    if result["success"]:
        print(f"Columns: {result['columns']}")
        print("Rows:")
        for row in result["rows"]:
            print(f"  {row}")
    else:
        print(f"Error: {result['error']}")
    
    # Test 4: Test forbidden query
    print("\n=== Test Forbidden Query ===")
    result = db.execute_query("DROP TABLE Products")
    print(f"Success: {result['success']}, Error: {result['error']}")
    
    # Test 5: Test invalid SQL
    print("\n=== Test Invalid SQL ===")
    result = db.execute_query("SELECT * FROM NonExistentTable")
    print(f"Success: {result['success']}, Error: {result['error']}")
    
    db.close()