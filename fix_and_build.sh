#!/bin/bash

# Fix and build QuantumLeapValidator project

echo "🔧 Fixing Xcode project references..."

# Reset to clean state
git checkout HEAD -- QuantumLeapValidator.xcodeproj/project.pbxproj

# Add missing Swift files to the project using xcodebuild
echo "📱 Adding UnifiedPerception files to project..."

# Create a temporary project file with the missing references
python3 << 'EOF'
import re

# Read the project file
with open('QuantumLeapValidator.xcodeproj/project.pbxproj', 'r') as f:
    content = f.read()

# Add file references for UnifiedPerception files
unified_files = [
    'UnifiedPerceptionBridge.swift',
    'UnifiedPerceptionView.swift', 
    'UnifiedPerceptionManager.swift'
]

# Find the PBXFileReference section and add our files
file_ref_section = re.search(r'(/\* Begin PBXFileReference section \*/.*?/\* End PBXFileReference section \*/)', content, re.DOTALL)
if file_ref_section:
    # Add file references before the end
    new_refs = []
    for i, filename in enumerate(unified_files):
        uuid = f"U{i}U{i}U{i}U{i}567890123456789ABC{i}"
        ref = f'\t\t{uuid} /* {filename} */ = {{isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = {filename}; sourceTree = "<group>"; }};'
        new_refs.append(ref)
    
    # Insert before the end comment
    end_pos = content.find('/* End PBXFileReference section */')
    content = content[:end_pos] + '\n'.join(new_refs) + '\n\t\t' + content[end_pos:]

# Find the QuantumLeapValidator group and add our files
group_section = re.search(r'(F691AD1DBDB5C76A0A5C65A0 /\* QuantumLeapValidator \*/ = \{.*?children = \(.*?\);)', content, re.DOTALL)
if group_section:
    # Add file references to the group
    children_end = content.find(');', group_section.end() - 20)
    for i, filename in enumerate(unified_files):
        uuid = f"U{i}U{i}U{i}U{i}567890123456789ABC{i}"
        ref = f'\t\t\t\t{uuid} /* {filename} */,\n'
        content = content[:children_end] + ref + content[children_end:]
        children_end += len(ref)

# Find the Sources build phase and add our files
sources_section = re.search(r'(0C627A6F6F4C07A91CC6DDBD /\* Sources \*/ = \{.*?files = \(.*?\);)', content, re.DOTALL)
if sources_section:
    # Add build file references
    files_end = content.find(');', sources_section.end() - 20)
    for i, filename in enumerate(unified_files):
        build_uuid = f"B{i}B{i}B{i}B{i}567890123456789DEF{i}"
        file_uuid = f"U{i}U{i}U{i}U{i}567890123456789ABC{i}"
        ref = f'\t\t\t\t{build_uuid} /* {filename} in Sources */,\n'
        content = content[:files_end] + ref + content[files_end:]
        files_end += len(ref)
    
    # Add PBXBuildFile entries
    build_file_section = re.search(r'(/\* Begin PBXBuildFile section \*/.*?/\* End PBXBuildFile section \*/)', content, re.DOTALL)
    if build_file_section:
        build_refs = []
        for i, filename in enumerate(unified_files):
            build_uuid = f"B{i}B{i}B{i}B{i}567890123456789DEF{i}"
            file_uuid = f"U{i}U{i}U{i}U{i}567890123456789ABC{i}"
            ref = f'\t\t{build_uuid} /* {filename} in Sources */ = {{isa = PBXBuildFile; fileRef = {file_uuid} /* {filename} */; }};'
            build_refs.append(ref)
        
        end_pos = content.find('/* End PBXBuildFile section */')
        content = content[:end_pos] + '\n'.join(build_refs) + '\n\t\t' + content[end_pos:]

# Write the updated project file
with open('QuantumLeapValidator.xcodeproj/project.pbxproj', 'w') as f:
    f.write(content)

print("✅ Project file updated with UnifiedPerception files")
EOF

echo "🏗️ Building project..."
xcodebuild -project QuantumLeapValidator.xcodeproj -scheme QuantumLeapValidator -destination 'platform=iOS Simulator,name=iPhone 16,OS=18.6' build

if [ $? -eq 0 ]; then
    echo "✅ Build successful! App is ready for testing."
else
    echo "❌ Build failed. Check the errors above."
fi
