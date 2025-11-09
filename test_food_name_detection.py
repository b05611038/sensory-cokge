#!/usr/bin/env python3
"""
Test script to verify food_name auto-detection in generate_synthetic_data
"""

from sensory_cokge import build_graph_from_hierarchy, generate_synthetic_data

def test_food_name_detection():
    """Test that food_name is correctly auto-detected from graph.graph_name"""

    print("=" * 60)
    print("Testing food_name auto-detection in generate_synthetic_data")
    print("=" * 60)

    # Test 1: Wine graph
    print("\nTest 1: Wine graph (should auto-detect 'wine' from graph_name)")
    wine_hierarchy = {
        'fruity': ['apple', 'pear', 'citrus'],
        'floral': ['rose', 'violet'],
        'spicy': ['pepper', 'cinnamon']
    }
    wine_graph = build_graph_from_hierarchy(wine_hierarchy, root='wine_root',
                                            graph_name='wine_flavor_wheel')

    wine_data = generate_synthetic_data(
        train_samples=10,
        eval_samples=5,
        graph=wine_graph,
        save_csv=False
    )

    # Check that the generated text contains "wine" not "coffee"
    sample_text = wine_data['train'][0]['selections'][0]
    print(f"Sample text: {sample_text[:100]}...")

    if 'wine' in sample_text.lower():
        print("✓ PASS: Text contains 'wine'")
    else:
        print("✗ FAIL: Text does not contain 'wine'")

    if 'coffee' in sample_text.lower():
        print("✗ FAIL: Text incorrectly contains 'coffee'")
    else:
        print("✓ PASS: Text does not contain 'coffee'")

    # Test 2: Cheese graph
    print("\nTest 2: Cheese graph (should auto-detect 'cheese' from graph_name)")
    cheese_hierarchy = {
        'dairy': ['milk', 'cream', 'butter'],
        'savory': ['umami', 'salty'],
        'pungent': ['funky', 'sharp']
    }
    cheese_graph = build_graph_from_hierarchy(cheese_hierarchy, root='cheese_root',
                                              graph_name='cheese_attributes')

    cheese_data = generate_synthetic_data(
        train_samples=10,
        eval_samples=5,
        graph=cheese_graph,
        save_csv=False
    )

    sample_text = cheese_data['train'][0]['selections'][0]
    print(f"Sample text: {sample_text[:100]}...")

    if 'cheese' in sample_text.lower():
        print("✓ PASS: Text contains 'cheese'")
    else:
        print("✗ FAIL: Text does not contain 'cheese'")

    # Test 3: Custom override
    print("\nTest 3: Wine graph with explicit override (should use 'red wine')")
    custom_data = generate_synthetic_data(
        train_samples=10,
        eval_samples=5,
        graph=wine_graph,
        food_name='red wine',
        save_csv=False
    )

    sample_text = custom_data['train'][0]['selections'][0]
    print(f"Sample text: {sample_text[:100]}...")

    if 'red wine' in sample_text.lower():
        print("✓ PASS: Text contains 'red wine'")
    else:
        print("✗ FAIL: Text does not contain 'red wine'")

    # Test 4: Complex graph_name patterns
    print("\nTest 4: Complex graph_name patterns")
    test_cases = [
        ('coffee_flavor_wheel(unduplicated)', 'coffee'),
        ('chocolate_taste_profile', 'chocolate'),
        ('beer-attributes', 'beer'),
        ('tea_aromas_v2', 'tea'),
    ]

    for graph_name_test, expected_food in test_cases:
        hierarchy = {'fruity': ['apple'], 'spicy': ['pepper']}
        test_graph = build_graph_from_hierarchy(hierarchy, root='root',
                                                graph_name=graph_name_test)
        test_data = generate_synthetic_data(
            train_samples=5,
            eval_samples=2,
            graph=test_graph,
            save_csv=False
        )
        sample = test_data['train'][0]['selections'][0]
        if expected_food in sample.lower():
            print(f"  ✓ '{graph_name_test}' → '{expected_food}' (PASS)")
        else:
            print(f"  ✗ '{graph_name_test}' → expected '{expected_food}' but not found (FAIL)")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == '__main__':
    test_food_name_detection()
