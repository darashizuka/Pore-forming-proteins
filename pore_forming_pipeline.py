import subprocess
import sys
import os

def run_script(script_name):
    """Execute a Python script"""
    if os.path.exists(script_name):
        print(f"\nRunning: {script_name}")
        try:
            subprocess.run([sys.executable, script_name], check=True)
            print(f"✓ {script_name} completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running {script_name}: {e}")
    else:
        print(f"✗ Script {script_name} not found!")

def get_feature_choice():
    """Ask user to choose feature type"""
    print("\n" + "="*50)
    print("FEATURE SELECTION")
    print("="*50)
    print("1. AA  - Amino Acid Composition")
    print("2. DP  - Dipeptide Composition") 
    print("3. Physico - Physico-chemical Properties")
    print("4. PseAAC - Pseudo Amino Acid Composition")
    print("5. ESM - ESM Embeddings")
    print("6. ProtBERT - ProtBERT Embeddings")
    print("7. All basic features (No embeddings)")
    
    while True:
        choice = input("\nSelect feature type (1-7): ").strip()
        if choice == '1':
            return 'aa'
        elif choice == '2':
            return 'dp'
        elif choice == '3':
            return 'physico'
        elif choice == '4':
            return 'pseaac'
        elif choice == '5':
            return 'esm'
        elif choice == '6':
            return 'protbert'
        elif choice == '7':
            return 'all_basic'
        else:
            print("Invalid choice! Please enter 1-7.")

def get_model_choice():
    """Ask user to choose model type"""
    print("\n" + "="*50)
    print("MODEL SELECTION")
    print("="*50)
    print("1. SVM")
    print('2. Random forest')
    print("3. XG Boost")
    print("4. ANN")

    
    while True:
        choice = input("\nSelect model type (1-4): ").strip()
        if choice == '1':
            return 'svm'
        elif choice == '2':
            return 'rf'
        elif choice == '3':
            return 'xg'
        elif choice == '4':
            return 'ann'
        else:
            print("Invalid choice! Please enter 1-4.")

def get_script_to_run(feature, model):

    
    # Define the mapping of feature+model combinations to scripts
    script_mapping = {
        # AA (Amino Acid) combinations
        ('aa', 'ann'): 'dl_build_model_ann.py',
        ('aa','rf'): 'dl_build_model_rf.py',
        ('aa', 'xg'): 'dl_build_model_xg.py',
        ('aa','svm'): 'dl_build_model_svm.py',
        
        
        # DP (Dipeptide) combinations
        ('dp', 'ann'): 'dl_build_model_ann.py',
        ('dp', 'rf'): 'dl_build_model_rf.py',
        ('dp', 'xg'): 'dl_build_model_xg.py',
        ('dp', 'svm'): 'dl_build_model_svm.py',
      
        # Physico-chemical combinations
        ('physico', 'rf'): 'physico_basic.py',
        ('physico', 'svm'): 'physico_basic.py',
        ('physico', 'xg'): 'physico_basic.py',
        ('physico', 'ann'): 'physico_ann.py',
        
        # PseAAC combinations
        ('pseaac', 'ann'): 'pseaac_ann.py',
        ('pseaac', 'rf'): 'pseaac_basic.py',
        ('pseaac', 'svm'): 'pseaac_basic.py',
        ('pseaac', 'xg'): 'pseaac_basic.py',

        # ESM Embeddings combinations
        ('esm', 'rf'): 'dl_dataset7.py',
        ('esm', 'svm'):'dl_dataset7.py',
        ('esm','xg'): 'dl_dataset7.py',
        ('esm','ann'): 'dl_dataset19.py',



        # ProtBERT combinations
        ('protbert', 'rf'): 'dl_dataset9.py',
        ('protbert', 'svm'):'dl_dataset9py',
        ('protbert','xg'): 'dl_dataset9.py',
        ('protbert','ann'): 'dl_dataset18.py',

        # All basic features combinations
        ('all_basic', 'rf'): 'dl_dataset14.py',
        ('all_basic', 'svm'):'dl_dataset14.py',
        ('all_basic', 'xg'):'dl_dataset14.py',
        ('all_basic', 'ann'):'dl_dataset16.py',

       
    }


    
    return script_mapping.get((feature, model))

def main():
    
    try:
        # Step 1: Get feature choice
        feature = get_feature_choice()
        print(f"\n✓ Selected feature: {feature.upper()}")
        
        # Step 2: Get model choice
        model = get_model_choice()
        print(f"\n✓ Selected model: {model.upper()}")
        
        # Step 3: Determine and run the appropriate script
        script_to_run = get_script_to_run(feature, model)
        
        if script_to_run:
            print(f"\n" + "="*50)
            print("RUNNING ANALYSIS")
            print("="*50)
            print(f"Feature: {feature.upper()}")
            print(f"Model: {model.upper()}")
            print(f"Script: {script_to_run}")
            print("="*50)
            
            # Confirm before running
            confirm = input(f"\nProceed with running {script_to_run}? (y/n): ").lower().strip()
            if confirm in ['y', 'yes']:
                run_script(script_to_run)
            else:
                print("Analysis cancelled.")
        else:
            print(f"\n✗ No script found for combination: {feature} + {model}")
            
            
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()