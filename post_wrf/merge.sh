#!/bin/bash

# Script function: Merge WRF output files based on parameter options, supporting time range, variable selection and coordinate variable handling
# Author：Evan
# Date：2025-05-28
# Dependencies: NCO toolkit (ncrcat, ncks) - installed in conda nco environment
# Usage: ./merge_from_input.sh < nco.inp

# 默认参数
# Default parameters
wrfout_directory="."
start_time=""
end_time=""
variables=""
domain="d01"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/data8/xuyf/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data8/xuyf/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/data8/xuyf/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/data8/xuyf/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate nco

# 读取标准输入并解析参数
# Read from stdin and parse parameters
while IFS= read -r line; do
    # 跳过空行和注释行
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    
    # 解析键值对
    # Parse key-value pairs
    if [[ "$line" =~ ^[[:space:]]*([^=]+)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        
        # 去除键和值的前后空格
        # Trim whitespace from key and value
        key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        # 根据键设置相应变量
        # Set variables based on key
        case "$key" in
            "wrfout_directory")
                wrfout_directory="$value"
                ;;
            "start_time")
                start_time="$value"
                ;;
            "end_time")
                end_time="$value"
                ;;
            "domain")
                # 处理domain格式，如果是数字则转换为d0X格式
                # Handle domain format, convert number to d0X format if needed
                if [[ "$value" =~ ^[0-9]+$ ]]; then
                    domain="d$(printf "%02d" "$value")"
                else
                    domain="$value"
                fi
                ;;
            "variables")
                # 清理变量列表中的空格
                # Clean up spaces in variable list
                variables=$(echo "$value" | sed 's/[[:space:]]*,[[:space:]]*/,/g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                ;;
            *)
                echo "Warning: Unknown parameter '$key' ignored"
                ;;
        esac
    fi
done

# 记录开始时间
# Record start time
script_start_time=$(date +%s)

echo "=========================================="
echo "WRF Data Merging Process Started"
echo "=========================================="
echo "WRF output directory: $wrfout_directory"
echo "Domain: $domain"
echo "Variables: ${variables:-'all variables'}"
if [[ -n "$start_time" && -n "$end_time" ]]; then
    echo "Time range: $start_time to $end_time (local time)"
else
    echo "Time range: all available files"
fi
echo ""

# -------------------------------------------------------------------
# 步骤 1：激活conda nco环境
# Step 1: Activate conda nco environment
# -------------------------------------------------------------------
# 检查NCO工具是否可用
# Check if NCO tools are available
command -v ncrcat >/dev/null 2>&1 || { 
    echo "Error: ncrcat command not found, please ensure NCO toolkit is installed"
    exit 1
}
command -v ncks >/dev/null 2>&1 || { 
    echo "Error: ncks command not found, please ensure NCO toolkit is installed"
    exit 1
}

# -------------------------------------------------------------------
# 步骤 2：时区转换函数（如果需要）
# Step 2: Time zone conversion function (if needed)
# -------------------------------------------------------------------
convert_to_utc() {
    local local_time="$1"
    date -d "${local_time} +0800" -u +"%Y-%m-%d_%H:%M:%S" 2>/dev/null || {
        echo "Error: Invalid time format: ${local_time}"
        exit 1
    }
}

# -------------------------------------------------------------------
# 步骤 3：查找和筛选文件
# Step 3: Find and filter files
# -------------------------------------------------------------------
# echo "Searching for WRF output files in: $wrfout_directory"

# 检查目录是否存在
# Check if directory exists
if [[ ! -d "$wrfout_directory" ]]; then
    echo "Error: Directory $wrfout_directory does not exist"
    exit 1
fi

# 生成文件名模板
# Generate file name pattern
file_pattern="wrfout_${domain}_??????????_??:??:??"

# 查找所有匹配文件
# Find all matching files
all_files=()
while IFS= read -r file; do
    all_files+=("$file")
done < <(find "$wrfout_directory" -maxdepth 1 -name "$file_pattern" -print | sort)

if [[ ${#all_files[@]} -eq 0 ]]; then
    echo "Error: No WRF output files found matching pattern: $file_pattern"
    echo "Please check:"
    echo "1. Directory path: $wrfout_directory"
    echo "2. Domain: $domain"
    echo "3. File naming convention: wrfout_${domain}_YYYY-mm-dd_HH:MM:SS.nc"
    exit 1
fi

# echo "Found ${#all_files[@]} WRF output files for domain $domain"

# 如果指定了时间范围，进行时间筛选
# If time range is specified, filter by time
valid_files=()
if [[ -n "$start_time" && -n "$end_time" ]]; then
    # echo "Filtering files by time range..."
    
    # 转换本地时间为UTC
    # Convert local time to UTC
    utc_start=$(convert_to_utc "$start_time")
    utc_end=$(convert_to_utc "$end_time")
    
    echo "UTC time range: $utc_start to $utc_end"
    
    for file in "${all_files[@]}"; do
        # 从文件名提取UTC时间
        # Extract UTC time from filename
        file_time=$(basename "$file" | awk -F_ '{print $3"_"$4}')
        
        # 时间范围比较 - 使用字符串比较
        # Time range comparison - using string comparison
        if [[ "$file_time" > "$utc_start" || "$file_time" == "$utc_start" ]] && \
           [[ "$file_time" < "$utc_end" || "$file_time" == "$utc_end" ]]; then
            valid_files+=("$file")
        fi
    done
    
    if [[ ${#valid_files[@]} -eq 0 ]]; then
        echo "Error: No files found in the specified time range"
        echo "UTC range: $utc_start to $utc_end"
        exit 1
    fi
    
    echo "Found ${#valid_files[@]} files in the specified time range:"
else
    # 使用所有找到的文件
    # Use all found files
    valid_files=("${all_files[@]}")
    echo "Using all found files:"
fi

# 显示要合并的文件
# Display files to be merged
# for file in "${valid_files[@]}"; do
#     echo "  - $(basename "$file")"
# done

# -------------------------------------------------------------------
# 步骤 4：构建输出文件名
# Step 4: Build output filename
# -------------------------------------------------------------------
if [[ -n "$start_time" && -n "$end_time" ]]; then
    # 使用时间范围构建文件名
    # Build filename with time range
    start_str=$(echo "$start_time" | sed 's/[: -]//g' | cut -c1-10)
    end_str=$(echo "$end_time" | sed 's/[: -]//g' | cut -c1-10)
    output_file="wrfout_${domain}_${start_str}_to_${end_str}.nc"
else
    # 使用简单文件名
    # Use simple filename
    output_file="wrfout_${domain}_merged.nc"
fi

temp_merged="temp_merged_$(date +%s).nc"
temp_coord="temp_coord_$(date +%s).nc"

echo ""
echo "Output file: $output_file"

# -------------------------------------------------------------------
# 步骤 5：合并主数据（包括坐标变量）
# Step 5: Merge main data (including coordinate variables)
# -------------------------------------------------------------------
# echo "Merging data..."

# 构建ncrcat命令
# Build ncrcat command
if [[ -n "$variables" ]]; then
    # 如果指定了变量列表，同时提取指定变量和坐标变量，但排除XTIME
    # If variables are specified, extract specified variables and coordinate variables, but exclude XTIME
    echo "Merging specified variables: $variables"
    echo "Running: ncrcat -O -v \"${variables},XLONG,XLAT\" [${#valid_files[@]} files] -o \"$temp_merged\""
    ncrcat -O -v "${variables},XLONG,XLAT" "${valid_files[@]}" -o "$temp_merged" || {
        echo "Error: Failed to merge specified variables"
        echo "This might be due to:"
        echo "1. Variable names not found in the files"
        echo "2. NCO version compatibility issues"
        echo "3. File format problems"
        echo "Try checking available variables with: ncdump -h [filename]"
        exit 1
    }
    
    # 移除XTIME变量（如果存在）
    # Remove XTIME variable (if exists)
    # echo "Removing XTIME variable from merged file..."
    temp_no_xtime="temp_no_xtime_$(date +%s).nc"
    ncks -O -C -x -v XTIME "$temp_merged" "$temp_no_xtime" 2>/dev/null || cp "$temp_merged" "$temp_no_xtime"
    mv "$temp_no_xtime" "$temp_merged"
    
    # 将坐标变量转换为2D（移除时间维度）
    # Convert coordinate variables to 2D (remove time dimension)
    # echo "Converting coordinate variables to 2D (removing time dimension)..."
    temp_coord_2d="temp_coord_2d_$(date +%s).nc"
    ncks -O -v XLONG,XLAT -d Time,0 "$temp_merged" "$temp_coord_2d" || {
        echo "Error: Failed to extract coordinate variables"
        exit 1
    }
    
    # 从合并文件中移除3D坐标变量
    # Remove 3D coordinate variables from merged file
    # echo "Removing 3D coordinate variables from merged file..."
    temp_no_coord="temp_no_coord_$(date +%s).nc"
    ncks -O -C -x -v XLONG,XLAT "$temp_merged" "$temp_no_coord" || {
        echo "Error: Failed to remove 3D coordinate variables"
        exit 1
    }
    
    # 将2D坐标变量转换为真正的2D（移除时间维度）
    # Convert 2D coordinate variables to true 2D (remove time dimension)
    temp_coord_final="temp_coord_final_$(date +%s).nc"
    ncwa -O -a Time "$temp_coord_2d" "$temp_coord_final" || {
        echo "Error: Failed to convert coordinate variables to 2D"
        exit 1
    }
    
    # 将2D坐标变量添加回文件
    # Add 2D coordinate variables back to file
    # echo "Adding 2D coordinate variables back to merged file..."
    ncks -A "$temp_coord_final" "$temp_no_coord" || {
        echo "Error: Failed to add 2D coordinate variables"
        exit 1
    }
    
    # 移除变量coordinates属性中对XTIME的引用
    # Remove XTIME references from coordinates attributes
    # echo "Removing XTIME references from coordinates attributes..."
    temp_clean_coords="temp_clean_coords_$(date +%s).nc"
    cp "$temp_no_coord" "$temp_clean_coords"
    
    # 使用ncatted移除coordinates属性中的XTIME引用
    # Use ncatted to remove XTIME references from coordinates attributes
    IFS=',' read -ra VAR_ARRAY <<< "$variables"
    for var in "${VAR_ARRAY[@]}"; do
        var=$(echo "$var" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')  # 去除前后空格
        # 检查变量是否存在coordinates属性
        if ncks -m "$temp_clean_coords" | grep -q "^${var}.*coordinates"; then
            # 获取当前coordinates属性值并移除XTIME引用
            coords_attr=$(ncks -m "$temp_clean_coords" | grep "^${var}.*coordinates" | sed 's/.*coordinates = "\([^"]*\)".*/\1/')
            new_coords=$(echo "$coords_attr" | sed 's/XTIME[[:space:]]*//g' | sed 's/[[:space:]]*XTIME//g' | sed 's/[[:space:]]\+/ /g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            
            if [[ -n "$new_coords" ]]; then
                ncatted -O -a coordinates,"${var}",o,c,"$new_coords" "$temp_clean_coords" 2>/dev/null || true
            else
                ncatted -O -a coordinates,"${var}",d,, "$temp_clean_coords" 2>/dev/null || true
            fi
        fi
    done
    
    # 重命名最终文件
    # Rename final file
    mv "$temp_clean_coords" "$output_file" || {
        echo "Error: Failed to rename output file"
        exit 1
    }
    
else
    # 如果未指定变量，合并所有变量但排除坐标变量和XTIME，然后添加2D坐标变量
    # If no variables specified, merge all variables but exclude coordinate variables and XTIME, then add 2D coordinates
    echo "Merging all variables"
    echo "Running: ncrcat -O -C -x -v XLONG,XLAT,XTIME [${#valid_files[@]} files] -o \"$temp_merged\""
    ncrcat -O -C -x -v XLONG,XLAT,XTIME "${valid_files[@]}" -o "$temp_merged" || {
        echo "Error: Failed to merge all variables"
        echo "This might be due to:"
        echo "1. File format incompatibility"
        echo "2. NCO version issues"
        echo "3. Insufficient disk space"
        exit 1
    }
    
    # 提取2D坐标变量
    # Extract 2D coordinate variables
    # echo "Extracting 2D coordinate variables from first file..."
    first_file="${valid_files[0]}"
    temp_coord="temp_coord_$(date +%s).nc"
    ncks -O -v XLONG,XLAT -d Time,0 "$first_file" "$temp_coord" || {
        echo "Error: Failed to extract coordinate variables"
        rm -f "$temp_merged"
        exit 1
    }
    
    # 转换为2D
    # Convert to 2D
    temp_coord_2d="temp_coord_2d_$(date +%s).nc"
    ncwa -O -a Time "$temp_coord" "$temp_coord_2d" || {
        echo "Error: Failed to convert coordinate variables to 2D"
        rm -f "$temp_merged" "$temp_coord"
        exit 1
    }
    
    # 添加2D坐标变量
    # Add 2D coordinate variables
    # echo "Adding 2D coordinate variables to merged file..."
    ncks -A "$temp_coord_2d" "$temp_merged" || {
        echo "Error: Failed to add coordinate variables"
        rm -f "$temp_merged" "$temp_coord_2d"
        exit 1
    }
    
    # 移除所有变量coordinates属性中对XTIME的引用
    # Remove XTIME references from all variables' coordinates attributes
    # echo "Removing XTIME references from coordinates attributes..."
    temp_clean_coords="temp_clean_coords_$(date +%s).nc"
    cp "$temp_merged" "$temp_clean_coords"
    
    # 获取所有变量名并处理coordinates属性
    # Get all variable names and process coordinates attributes
    all_vars=$(ncks -m "$temp_clean_coords" | grep -E "^[a-zA-Z].*coordinates" | cut -d':' -f1 | sort -u)
    for var in $all_vars; do
        # 检查变量是否存在coordinates属性
        if ncks -m "$temp_clean_coords" | grep -q "^${var}.*coordinates"; then
            # 获取当前coordinates属性值并移除XTIME引用
            coords_attr=$(ncks -m "$temp_clean_coords" | grep "^${var}.*coordinates" | sed 's/.*coordinates = "\([^"]*\)".*/\1/')
            new_coords=$(echo "$coords_attr" | sed 's/XTIME[[:space:]]*//g' | sed 's/[[:space:]]*XTIME//g' | sed 's/[[:space:]]\+/ /g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            
            if [[ -n "$new_coords" ]]; then
                ncatted -O -a coordinates,"${var}",o,c,"$new_coords" "$temp_clean_coords" 2>/dev/null || true
            else
                ncatted -O -a coordinates,"${var}",d,, "$temp_clean_coords" 2>/dev/null || true
            fi
        fi
    done
    
    # 重命名最终文件
    # Rename final file
    mv "$temp_clean_coords" "$output_file" || {
        echo "Error: Failed to rename output file"
        exit 1
    }
fi

# echo "Data merging completed successfully."

# -------------------------------------------------------------------
# 步骤 8：清理临时文件和输出结果
# Step 8: Clean up temporary files and output results
# -------------------------------------------------------------------
echo "Cleaning up temporary files..."
rm -f temp_merged_*.nc temp_coord_*.nc temp_no_coord_*.nc temp_coord_final_*.nc temp_clean_coords_*.nc temp_no_xtime_*.nc

# 计算运行时间
# Calculate runtime
script_end_time=$(date +%s)
duration=$((script_end_time - script_start_time))

# Output results
# Verify output file
if [[ -f "$output_file" ]]; then
    echo ""
    echo "=========================================="
    echo "WRF data merging completed successfully!"
    echo "=========================================="
    echo "Output file: $(realpath "$output_file")"
    echo "File size: $(du -h "$output_file" | cut -f1)"
    echo "Files merged: ${#valid_files[@]}"
    echo "Source directory: $wrfout_directory"
    if [[ -n "$start_time" && -n "$end_time" ]]; then
        echo "Time range: $start_time to $end_time (local)"
        echo "UTC range: $utc_start to $utc_end"
    else
        echo "Time range: all available files"
    fi
    echo "Domain: $domain"
    echo "Variables: ${variables:-'all variables (with coordinate fix and XTIME removal)'}"
    printf "Runtime: %02d:%02d:%02d (HH:MM:SS)\n" $((duration/3600)) $((duration%3600/60)) $((duration%60))
    # echo "Note: XTIME variable has been removed and coordinates attributes cleaned for ncview compatibility"
    echo "=========================================="
else
    echo "File verification: FAILED"
    printf "Runtime: %02d:%02d:%02d (HH:MM:SS)\n" $((duration/3600)) $((duration%3600/60)) $((duration%60))
    exit 1
fi 