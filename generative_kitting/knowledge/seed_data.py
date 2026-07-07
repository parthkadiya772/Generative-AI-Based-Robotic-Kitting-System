"""
Seed data for the Kitting Parts Database.

Populates the database with sample parts matching the USD meshes
in the workspace, and defines example kit configurations.
"""

from knowledge.parts_database import PartsDatabase
from utils.logger import log


def seed_parts(db: PartsDatabase) -> None:
    """
    Insert all known industrial parts into the database.
    Part metadata matches the USD meshes in ``usd_meshes/``.
    """
    parts = [
        {
            "part_id": "motor_valve_001",
            "label": "motor_valve",
            "category": "actuator",
            "weight_grams": 320.0,
            "material": "steel",
            "grip_type": "parallel",
            "grip_force_n": 45.0,
            "fragile": False,
            "stackable": False,
            "max_stack_height": 1,
            "usd_mesh_file": "motor_valve.usd",
            "description": "Grey steel motor valve assembly, ~80mm diameter",
        },
        {
            "part_id": "black_hose_001",
            "label": "black_hose",
            "category": "tubing",
            "weight_grams": 85.0,
            "material": "rubber",
            "grip_type": "parallel",
            "grip_force_n": 25.0,
            "fragile": False,
            "stackable": True,
            "max_stack_height": 5,
            "usd_mesh_file": "black_hose.usd",
            "description": "Flexible black rubber hose, ~150mm length",
        },
        {
            "part_id": "black_plate_001",
            "label": "black_plate",
            "category": "structural",
            "weight_grams": 210.0,
            "material": "steel",
            "grip_type": "parallel",
            "grip_force_n": 40.0,
            "fragile": False,
            "stackable": True,
            "max_stack_height": 10,
            "usd_mesh_file": "black_plate.usd",
            "description": "Flat black steel plate, ~100x60mm",
        },
        {
            "part_id": "black_plug_001",
            "label": "black_plug",
            "category": "connector",
            "weight_grams": 45.0,
            "material": "plastic",
            "grip_type": "parallel",
            "grip_force_n": 20.0,
            "fragile": True,
            "stackable": False,
            "max_stack_height": 1,
            "usd_mesh_file": "black_plug.usd",
            "description": "Black plastic electrical plug connector",
        },
        {
            "part_id": "small_hinge_001",
            "label": "small_hinge",
            "category": "hardware",
            "weight_grams": 35.0,
            "material": "steel",
            "grip_type": "parallel",
            "grip_force_n": 30.0,
            "fragile": False,
            "stackable": True,
            "max_stack_height": 8,
            "usd_mesh_file": "small_hinge.usd",
            "description": "Small silver steel hinge, ~40mm",
        },
        {
            "part_id": "small_tube_001",
            "label": "small_tube",
            "category": "tubing",
            "weight_grams": 60.0,
            "material": "aluminum",
            "grip_type": "parallel",
            "grip_force_n": 25.0,
            "fragile": False,
            "stackable": True,
            "max_stack_height": 6,
            "usd_mesh_file": "small_tube.usd",
            "description": "Small aluminum tube, ~80mm length",
        },
        {
            "part_id": "silver_box_001",
            "label": "silver_box",
            "category": "housing",
            "weight_grams": 180.0,
            "material": "aluminum",
            "grip_type": "parallel",
            "grip_force_n": 35.0,
            "fragile": False,
            "stackable": True,
            "max_stack_height": 4,
            "usd_mesh_file": "silver_box.usd",
            "description": "Small silver aluminum enclosure box, ~60x40x30mm",
        },
        {
            "part_id": "silver_gun_001",
            "label": "silver_gun",
            "category": "tool",
            "weight_grams": 450.0,
            "material": "steel",
            "grip_type": "parallel",
            "grip_force_n": 50.0,
            "fragile": False,
            "stackable": False,
            "max_stack_height": 1,
            "usd_mesh_file": "silver_gun.usd",
            "description": "Silver steel pneumatic dispensing gun assembly",
        },
        {
            "part_id": "tube_with_clamps_001",
            "label": "tube_with_clamps",
            "category": "assembly",
            "weight_grams": 150.0,
            "material": "steel",
            "grip_type": "parallel",
            "grip_force_n": 40.0,
            "fragile": False,
            "stackable": False,
            "max_stack_height": 1,
            "usd_mesh_file": "tube_with_clamps.usd",
            "description": "Steel tube with attached hose clamps, ~120mm length",
        },
    ]

    for p in parts:
        db.add_part(**p)
    log.info(f"Seeded {len(parts)} parts into database")


def seed_kits(db: PartsDatabase) -> None:
    """Insert example kit definitions."""
    kits = [
        {
            "kit_id": "kit_motor_assembly",
            "kit_name": "Motor Assembly Kit",
            "required_parts": [
                {"label": "motor_valve", "quantity": 1},
                {"label": "black_hose", "quantity": 2},
                {"label": "tube_with_clamps", "quantity": 1},
                {"label": "black_plug", "quantity": 2},
            ],
            "description": "Complete motor valve assembly with tubing and connectors",
        },
        {
            "kit_id": "kit_connector_set",
            "kit_name": "Connector Kit",
            "required_parts": [
                {"label": "black_plug", "quantity": 4},
                {"label": "small_hinge", "quantity": 2},
                {"label": "small_tube", "quantity": 2},
            ],
            "description": "Electrical and mechanical connector set",
        },
        {
            "kit_id": "kit_structural",
            "kit_name": "Structural Components Kit",
            "required_parts": [
                {"label": "black_plate", "quantity": 2},
                {"label": "silver_box", "quantity": 1},
                {"label": "small_hinge", "quantity": 4},
            ],
            "description": "Plates, enclosures, and mounting hardware",
        },
    ]

    for k in kits:
        db.add_kit(**k)
    log.info(f"Seeded {len(kits)} kit definitions into database")


def seed_database(db_path: str = "knowledge/kitting.db") -> PartsDatabase:
    """
    Create and populate a fresh database with sample data.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.

    Returns
    -------
    PartsDatabase
        The populated database instance.
    """
    db = PartsDatabase(db_path)
    seed_parts(db)
    seed_kits(db)
    log.info("Database seeding complete")
    return db


if __name__ == "__main__":
    db = seed_database()
    print(f"\nParts: {len(db.get_all_parts())}")
    print(f"Kits:  {len(db.get_all_kits())}")
    for p in db.get_all_parts():
        print(f"  {p['part_id']:30s} {p['label']:20s} {p['category']}")
    for k in db.get_all_kits():
        print(f"  {k['kit_id']:30s} {k['kit_name']:25s} parts={k['required_parts']}")
    db.close()
