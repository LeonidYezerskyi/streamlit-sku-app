import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SKU Portfolio Dashboard", page_icon="📊", layout="wide")

# Initialize session state for bookmarked SKUs
if 'bookmarked_skus' not in st.session_state:
    st.session_state.bookmarked_skus = set()

@st.cache_data
def load_sku_data():
    with open('input_data.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data).iloc[1:].reset_index(drop=True)
    
    column_mapping = {'': 'SKU', 'Brand': 'Brand_IV', '__1': 'Brand_IV_Prob', '__2': 'Brand_VaR10', 'Brand Owner': 'Owner_IV', '__3': 'Owner_IV_Prob', '__4': 'Owner_VaR10', 'Brand Source of Volume': 'Brand_Capture_Eff', '__5': 'Brand_Cannibal_Ratio', '__6': 'Brand_Category_Lift', '__7': 'Brand_Abs_Captured', '__8': 'Brand_Abs_Cannibalized', '__9': 'Brand_Abs_Category_Lift', 'Brand Owner Source of Volume': 'Owner_Capture_Eff', '__10': 'Owner_Cannibal_Ratio', '__11': 'Owner_Category_Lift', '__12': 'Owner_Abs_Captured', '__13': 'Owner_Abs_Cannibalized', '__14': 'Owner_Abs_Category_Lift', 'Mean Diversion Ratio': 'Top_Competitor', '__15': 'Top_Competitor_Ratio', '__16': 'Second_Competitor', '__17': 'Second_Competitor_Ratio', '__18': 'Third_Competitor', '__19': 'Third_Competitor_Ratio', 'Market Concentration (Diversion HHI)': 'Market_HHI'}
    df = df.rename(columns=column_mapping)
    
    percentage_columns = ['Brand_IV', 'Brand_IV_Prob', 'Brand_VaR10', 'Owner_IV', 'Owner_IV_Prob', 'Owner_VaR10', 'Brand_Capture_Eff', 'Brand_Cannibal_Ratio', 'Brand_Category_Lift', 'Brand_Abs_Captured', 'Brand_Abs_Cannibalized', 'Brand_Abs_Category_Lift', 'Owner_Capture_Eff', 'Owner_Cannibal_Ratio', 'Owner_Category_Lift', 'Owner_Abs_Captured', 'Owner_Abs_Cannibalized', 'Owner_Abs_Category_Lift', 'Top_Competitor_Ratio', 'Second_Competitor_Ratio', 'Third_Competitor_Ratio', 'Market_HHI']
    
    for col in percentage_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.rstrip('%').astype(float)
    
    df['Composite_Score'] = df['Brand_IV'] * 0.4 + df['Brand_Capture_Eff'] * 0.4 + (100 - df['Brand_Cannibal_Ratio']) * 0.2
    return df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)

def get_traffic_light_status(value, metric_type):
    if metric_type == 'IV':
        if value >= 6.0: return '🟢 Go'
        elif value >= 4.0: return '🟡 Watch'
        else: return '🔴 Hold'
    elif metric_type == 'Capture':
        if value >= 80: return '🎯🟢 Go'
        elif value >= 70: return '🎯🟡 Watch'
        else: return '🎯🔴 Hold'
    elif metric_type == 'Cannibal':
        if value <= 10: return '♻️🟢 Go'
        elif value <= 20: return '♻️🟡 Watch'
        else: return '♻️🔴 Hold'
    elif metric_type == 'VaR':
        if value <= 4.0: return '🛡️🟢 Go'
        elif value <= 6.0: return '🛡️🟡 Watch'
        else: return '🛡️🔴 Hold'
    elif metric_type == 'Category_Lift':
        if value >= 6.0: return '🌊🟢 High'
        elif value >= 4.0: return '🌊🟡 Moderate'
        elif value >= 2.0: return '🌊🟠 Low'
        else: return '🌊🔴 Minimal'
    else:
        return '⚪'

def get_why_tag(value, metric_type):
    if metric_type == 'IV':
        if value >= 6.0: return "💰 Strong IV"
        elif value >= 4.0: return "💰 Moderate IV"
        else: return "💰 Low IV"
    elif metric_type == 'Capture':
        if value >= 80: return "🎯 High capture"
        elif value >= 70: return "🎯 Good capture"
        else: return "🎯 Low capture"
    elif metric_type == 'Cannibal':
        if value <= 10: return "♻️ Low cannibal"
        elif value <= 20: return "♻️ Moderate cannibal"
        else: return "♻️ High cannibal"
    elif metric_type == 'VaR':
        if value <= 4.0: return "🛡️ Low risk"
        elif value <= 6.0: return "🛡️ Moderate risk"
        else: return "🛡️ High risk"
    elif metric_type == 'Category_Lift':
        if value >= 7.0: return "🌊 Strong lift"
        elif value >= 5.0: return "🌊 Moderate lift"
        else: return "🌊 Limited lift"
    elif metric_type == 'HHI':
        if value >= 6.5: return "🏪 High concentration"
        elif value >= 6.0: return "🏪 Moderate concentration"
        else: return "🏪 Low concentration"
    elif metric_type == 'Prob':
        if value >= 95: return "📊 Very high conf"
        elif value >= 90: return "📊 High conf"
        elif value >= 85: return "📊 Good conf"
        elif value >= 80: return "📊 Moderate conf"
        elif value >= 70: return "📊 Low conf"
        else: return "📊 Very low conf"
    else:
        return "📈 Standard"

def get_action_recommendation(row):
    """
    Generate action recommendation with confidence level based on key metrics
    """
    brand_iv = row['Brand_IV']
    capture_eff = row['Brand_Capture_Eff']
    cannibal_ratio = row['Brand_Cannibal_Ratio']
    var10 = row['Brand_VaR10']
    prob_iv = row['Brand_IV_Prob']
    
    # Calculate confidence score (0-100) - FIXED VERSION
    confidence_score = (
        min(brand_iv / 10 * 25, 25) +  # Max 25 points for IV (capped at 10%)
        min(capture_eff / 100 * 25, 25) +  # Max 25 points for capture efficiency
        min((100 - cannibal_ratio) / 100 * 20, 20) +  # Max 20 points for low cannibalization
        min((100 - var10) / 10 * 15, 15) +  # Max 15 points for low VaR
        min(prob_iv / 100 * 15, 15)  # Max 15 points for high probability
    )
    
    # Ensure score is between 0-100
    confidence_score = min(confidence_score, 100)
    
    # Determine recommendation and confidence
    if confidence_score >= 80:
        if brand_iv >= 6.0 and capture_eff >= 80:
            recommendation = "🚀 Launch with high confidence"
            confidence = "Very High"
            color = "success"
        else:
            recommendation = "✅ Launch with confidence"
            confidence = "High"
            color = "success"
    elif confidence_score >= 60:
        if cannibal_ratio >= 20:
            recommendation = "⚠️ Pilot test recommended"
            confidence = "Medium"
            color = "warning"
        else:
            recommendation = "📊 Launch with monitoring"
            confidence = "Medium"
            color = "warning"
    elif confidence_score >= 40:
        recommendation = "🔍 Pilot test first"
        confidence = "Low"
        color = "warning"
    else:
        recommendation = "⏸️ Defer launch"
        confidence = "Very Low"
        color = "error"
    
    return recommendation, confidence, confidence_score, color

def create_sku_banner(row):
    """
    Create simple top banner with SKU name only
    """
    banner_html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    ">
        <h2 style="margin: 0; font-size: 28px; font-weight: bold;">{row['SKU']}</h2>
    </div>
    """
    return banner_html

def create_compact_sku_card(row, rank):
    """
    Create compact SKU card for 4-up grid layout using Streamlit components
    """
    recommendation, confidence, confidence_score, rec_color = get_action_recommendation(row)
    
    # Create card using Streamlit components instead of HTML
    with st.container():
        # SKU header with bookmark - more compact
        col_header1, col_header2 = st.columns([4, 1])
        with col_header1:
            st.markdown(f"**{row['SKU']}** #{rank}")
        with col_header2:
            create_bookmark_button(row['SKU'], row)
        
        # Compact recommendation
        if rec_color == "success":
            st.success(confidence)
        elif rec_color == "warning":
            st.warning(confidence)
        else:
            st.error(confidence)
        
        # Compact metrics with in-cell bars
        st.markdown("**Metrics:**")
        st.caption(f"💰 IV: {create_in_cell_bar(row['Brand_IV'], 8, 6)}")
        st.caption(f"🎯 Capture: {create_in_cell_bar(row['Brand_Capture_Eff'], 100, 6)}")
        st.caption(f"♻️ Cannibal: {create_in_cell_bar(row['Brand_Cannibal_Ratio'], 30, 6, reverse=True)}")
        st.caption(f"🛡️ VaR: {create_in_cell_bar(row['Brand_VaR10'], 8, 6, reverse=True)}")
        
        # Compact score
        st.caption(f"Score: {row['Composite_Score']:.1f}")
        
        # Compact sensitivity indicator
        high_sensitivity_count = sum([
            row['Brand_IV'] >= 6.0,
            row['Brand_Capture_Eff'] >= 80,
            row['Brand_Cannibal_Ratio'] >= 15,
            row['Brand_VaR10'] >= 5
        ])
        
        if high_sensitivity_count >= 3:
            st.caption("🔴 High Sens")
        elif high_sensitivity_count >= 2:
            st.caption("🟡 Mod Sens")
        else:
            st.caption("🟢 Low Sens")

def get_risk_tile(row):
    var10 = row['Brand_VaR10']
    prob_iv = row['Brand_IV_Prob']
    
    risk_score = (var10 * 0.6) + ((100 - prob_iv) * 0.4)
    
    if risk_score <= 2.0:
        risk_level = "🛡️🟢 Low Risk"
        risk_color = "success"
        risk_desc = "Low VaR, High Probability"
    elif risk_score <= 4.0:
        risk_level = "🛡️🟡 Medium Risk"
        risk_color = "warning"
        risk_desc = "Moderate Risk Profile"
    else:
        risk_level = "🛡️🔴 High Risk"
        risk_color = "error"
        risk_desc = "High VaR, Low Probability"
    
    return risk_level, risk_color, risk_desc, risk_score

def get_net_effect(row):
    owner_iv = row['Owner_IV']
    owner_cannibal = row['Owner_Cannibal_Ratio']
    
    net_effect = owner_iv - (owner_cannibal * 0.2)
    
    if net_effect >= 2.0:
        return "♻️⬆️ Net Positive", "success", f"Strong positive impact on owner (+{net_effect:.1f}%)"
    elif net_effect >= 0.5:
        return "♻️➡️ Net Neutral", "warning", f"Moderate impact on owner (+{net_effect:.1f}%)"
    else:
        return "♻️⬇️ Net Negative", "error", f"Negative impact on owner ({net_effect:.1f}%)"

def create_sparkbar(df, current_value, metric_name, reverse=False):
    median_val = df[metric_name].median()
    q75_val = df[metric_name].quantile(0.75)
    q25_val = df[metric_name].quantile(0.25)
    
    # Create compact text-based bars with short labels
    if reverse:
        if current_value <= q25_val:
            return "██████████ Top 25%"
        elif current_value <= median_val:
            return "████████   Above Med"
        else:
            return "████      Below Med"
    else:
        if current_value >= q75_val:
            return "██████████ Top 25%"
        elif current_value >= median_val:
            return "████████   Above Med"
        else:
            return "████      Below Med"

def create_in_cell_bar(value, max_value, width=10, reverse=False):
    """
    Create a compact in-cell bar visualization
    """
    if max_value == 0:
        return "─" * width
    
    # Calculate bar length
    if reverse:
        bar_length = int((1 - value / max_value) * width)
    else:
        bar_length = int((value / max_value) * width)
    
    bar_length = max(0, min(width, bar_length))
    
    # Create bar with padding
    bar = "█" * bar_length + "░" * (width - bar_length)
    return f"{bar} {value:.1f}%"

def create_compact_metrics_table(df):
    """
    Create a compact metrics table with in-cell bars and short labels
    """
    st.markdown("### 📊 Compact Portfolio Metrics")
    
    # Calculate max values for normalization
    max_iv = df['Brand_IV'].max()
    max_capture = df['Brand_Capture_Eff'].max()
    max_cannibal = df['Brand_Cannibal_Ratio'].max()
    max_var = df['Brand_VaR10'].max()
    
    # Create compact table with in-cell bars
    for idx, row in df.iterrows():
        with st.expander(f"📊 {row['SKU']} - Score: {row['Composite_Score']:.1f}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**💰 IV**")
                st.code(create_in_cell_bar(row['Brand_IV'], max_iv, 8))
                st.caption(f"Rank: {df['Brand_IV'].rank(ascending=False)[idx]:.0f}")
            
            with col2:
                st.markdown("**🎯 Capture**")
                st.code(create_in_cell_bar(row['Brand_Capture_Eff'], max_capture, 8))
                st.caption(f"Rank: {df['Brand_Capture_Eff'].rank(ascending=False)[idx]:.0f}")
            
            with col3:
                st.markdown("**♻️ Cannibal**")
                st.code(create_in_cell_bar(row['Brand_Cannibal_Ratio'], max_cannibal, 8, reverse=True))
                st.caption(f"Rank: {df['Brand_Cannibal_Ratio'].rank(ascending=True)[idx]:.0f}")
            
            with col4:
                st.markdown("**🛡️ VaR**")
                st.code(create_in_cell_bar(row['Brand_VaR10'], max_var, 8, reverse=True))
                st.caption(f"Rank: {df['Brand_VaR10'].rank(ascending=True)[idx]:.0f}")

def create_compact_summary_cards(df):
    """
    Create compact summary cards with whitespace
    """
    st.markdown("### 📈 Portfolio Summary")
    
    # Top performers
    top_3 = df.head(3)
    
    col1, col2, col3 = st.columns(3)
    
    for i, (_, row) in enumerate(top_3.iterrows()):
        with [col1, col2, col3][i]:
            st.markdown(f"**#{i+1} {row['SKU']}**")
            st.metric("Score", f"{row['Composite_Score']:.1f}")
            
            # Compact metrics with bars
            st.markdown("**Key Metrics:**")
            st.caption(f"💰 {create_in_cell_bar(row['Brand_IV'], 8, 6)}")
            st.caption(f"🎯 {create_in_cell_bar(row['Brand_Capture_Eff'], 100, 6)}")
            st.caption(f"♻️ {create_in_cell_bar(row['Brand_Cannibal_Ratio'], 30, 6, reverse=True)}")
            
            # Recommendation
            recommendation, confidence, _, rec_color = get_action_recommendation(row)
            if rec_color == "success":
                st.success(confidence)
            elif rec_color == "warning":
                st.warning(confidence)
            else:
                st.error(confidence)

def create_competitor_panel(row):
    st.markdown("**🎯 Competitors**")
    
    competitors = [
        {'name': row['Top_Competitor'], 'ratio': row['Top_Competitor_Ratio'], 'rank': 1},
        {'name': row['Second_Competitor'], 'ratio': row['Second_Competitor_Ratio'], 'rank': 2},
        {'name': row['Third_Competitor'], 'ratio': row['Third_Competitor_Ratio'], 'rank': 3}
    ]
    
    # Compact competitor display
    for comp in competitors:
        if comp['ratio'] > 0:
            st.caption(f"#{comp['rank']} {comp['name']}: {comp['ratio']:.1f}%")

def create_mini_scatter_plot(df, current_sku):
    fig = go.Figure()
    
    for idx, row in df.iterrows():
        if row['SKU'] == current_sku:
            # Calculate confidence intervals for current SKU
            iv_confidence = row['Brand_IV_Prob'] / 100 * 0.3
            cannibal_confidence = (100 - row['Brand_IV_Prob']) / 100 * 0.2
            
            fig.add_trace(go.Scatter(
                x=[row['Brand_IV']],
                y=[row['Brand_Cannibal_Ratio']],
                mode='markers',
                marker=dict(
                    size=row['Brand_Capture_Eff'] * 2,
                    symbol='star',
                    color='white',
                    line=dict(width=3, color='black')
                ),
                name=row['SKU'],
                error_x=dict(
                    type='data',
                    array=[iv_confidence],
                    visible=True,
                    width=3,
                    color='rgba(0,0,0,0.4)',
                    thickness=1
                ),
                error_y=dict(
                    type='data',
                    array=[cannibal_confidence],
                    visible=True,
                    width=3,
                    color='rgba(0,0,0,0.4)',
                    thickness=1
                ),
                text=[f"{row['SKU']}<br>IV: {row['Brand_IV']:.1f}% ±{iv_confidence:.1f}%<br>Cannibal: {row['Brand_Cannibal_Ratio']:.1f}% ±{cannibal_confidence:.1f}%<br>Capture: {row['Brand_Capture_Eff']:.1f}%<br>Confidence: {row['Brand_IV_Prob']:.1f}%"],
                hovertemplate='%{text}<extra></extra>'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[row['Brand_IV']],
                y=[row['Brand_Cannibal_Ratio']],
                mode='markers',
                marker=dict(
                    size=row['Brand_Capture_Eff'] * 2,
                    symbol='circle',
                    color='lightgray',
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                name=row['SKU'],
                text=[f"{row['SKU']}<br>IV: {row['Brand_IV']:.1f}%<br>Cannibal: {row['Brand_Cannibal_Ratio']:.1f}%<br>Capture: {row['Brand_Capture_Eff']:.1f}%"],
                hovertemplate='%{text}<extra></extra>',
                showlegend=False
            ))
    
    # Standardized scales for consistent visual comparisons
    fig.update_layout(
        title=f"Portfolio Position - {current_sku}",
        xaxis_title="Incremental Value (%)",
        yaxis_title="Cannibalization Ratio (%)",
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        # Standardized scales
        xaxis=dict(
            range=[0, 8],  # Standard IV scale: 0-8%
            dtick=1,       # Tick every 1%
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        yaxis=dict(
            range=[0, 25], # Standard Cannibal scale: 0-25%
            dtick=5,       # Tick every 5%
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        plot_bgcolor='white'
    )
    
    return fig

def get_human_summary(row):
    if row['Brand_IV'] >= 6.0 and row['Brand_Capture_Eff'] >= 80:
        return "Strong incremental value with high capture efficiency - ready for launch"
    elif row['Brand_Cannibal_Ratio'] >= 20:
        return "High cannibalization risk - monitor internal impact carefully"
    elif row['Brand_Category_Lift'] >= 7.0:
        return "Strong category expansion potential - focus on market growth"
    else:
        return "Moderate performance - consider pilot testing first"

def create_source_volume_chart(row):
    brand_data = {
        'Captured': row['Brand_Abs_Captured'],
        'Cannibalized': row['Brand_Abs_Cannibalized'],
        'Category Lift': row['Brand_Abs_Category_Lift']
    }
    
    owner_data = {
        'Captured': row['Owner_Abs_Captured'],
        'Cannibalized': row['Owner_Abs_Cannibalized'],
        'Category Lift': row['Owner_Abs_Category_Lift']
    }
    
    fig = go.Figure()
    
    # Calculate confidence intervals for each metric
    # Using capture efficiency as proxy for confidence (higher capture = more confident)
    brand_confidence = row['Brand_Capture_Eff'] / 100 * 0.2  # 20% of capture efficiency as CI
    owner_confidence = row['Owner_Capture_Eff'] / 100 * 0.2
    
    fig.add_trace(go.Bar(
        name='Brand Captured',
        x=['Brand', 'Owner'],
        y=[brand_data['Captured'], owner_data['Captured']],
        marker_color='lightgray',
        marker_line=dict(color='black', width=1),
        text=[f"Captured: {brand_data['Captured']:.1f}%", f"Captured: {owner_data['Captured']:.1f}%"],
        textposition='inside',
        textfont=dict(color='black', size=10),
        error_y=dict(
            type='data',
            array=[brand_confidence, owner_confidence],
            visible=True,
            width=2,
            color='rgba(0,0,0,0.4)',
            thickness=1
        )
    ))
    
    fig.add_trace(go.Bar(
        name='Cannibalized',
        x=['Brand', 'Owner'],
        y=[brand_data['Cannibalized'], owner_data['Cannibalized']],
        marker_color='darkgray',
        marker_line=dict(color='black', width=1),
        text=[f"Cannibal: {brand_data['Cannibalized']:.1f}%", f"Cannibal: {owner_data['Cannibalized']:.1f}%"],
        textposition='inside',
        textfont=dict(color='white', size=10),
        error_y=dict(
            type='data',
            array=[brand_confidence * 0.5, owner_confidence * 0.5],  # Smaller CI for cannibalization
            visible=True,
            width=2,
            color='rgba(0,0,0,0.4)',
            thickness=1
        )
    ))
    
    fig.add_trace(go.Bar(
        name='Category Lift',
        x=['Brand', 'Owner'],
        y=[brand_data['Category Lift'], owner_data['Category Lift']],
        marker_color='white',
        marker_line=dict(color='black', width=2),
        text=[f"Lift: {brand_data['Category Lift']:.1f}%", f"Lift: {owner_data['Category Lift']:.1f}%"],
        textposition='inside',
        textfont=dict(color='black', size=10),
        error_y=dict(
            type='data',
            array=[brand_confidence * 0.8, owner_confidence * 0.8],  # Medium CI for category lift
            visible=True,
            width=2,
            color='rgba(0,0,0,0.4)',
            thickness=1
        )
    ))
    
    fig.update_layout(
        barmode='stack',
        title=f"Source of Volume - {row['SKU']}",
        height=300,
        showlegend=True,
        xaxis_title="",
        yaxis_title="Percentage (%)",
        margin=dict(l=20, r=20, t=40, b=20),
        # Standardized scale for volume charts
        yaxis=dict(
            range=[0, 10], # Standard volume scale: 0-10%
            dtick=2,       # Tick every 2%
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        plot_bgcolor='white'
    )
    
    return fig

def create_top5_launch_callout(df):
    """
    Create Top 5 to launch now callout using Streamlit components
    """
    # Get top 5 SKUs by composite score
    top5 = df.head(5)
    
    # Create callout using Streamlit components
    st.markdown("### 🚀 Top 5 to Launch Now")
    st.markdown("*Highest performing SKUs ready for immediate launch*")
    
    # Create 5 columns for top 5 SKUs
    cols = st.columns(5)
    
    for idx, (_, row) in enumerate(top5.iterrows()):
        recommendation, confidence, confidence_score, rec_color = get_action_recommendation(row)
        
        with cols[idx]:
            st.markdown(f"**{row['SKU']}**")
            st.metric("Score", f"{row['Composite_Score']:.1f}")
            
            # Recommendation with color
            if rec_color == "success":
                st.success(recommendation)
            elif rec_color == "warning":
                st.warning(recommendation)
            else:
                st.error(recommendation)
            
            st.caption(f"Confidence: {confidence}")
    

def create_portfolio_risk_map(df):
    """
    Create portfolio risk map charting all SKUs by VaR10 vs Incremental Value
    """
    fig = go.Figure()
    
    # Add scatter plot for all SKUs
    for idx, row in df.iterrows():
        recommendation, confidence, confidence_score, rec_color = get_action_recommendation(row)
        
        # Determine marker style based on recommendation (using shapes instead of colors)
        if rec_color == "success":
            marker_symbol = "circle"
            marker_line_color = "black"
        elif rec_color == "warning":
            marker_symbol = "square"
            marker_line_color = "black"
        else:
            marker_symbol = "diamond"
            marker_line_color = "black"
        
        # Add scatter point with confidence intervals
        # Calculate confidence intervals (simplified - using probability as confidence width)
        iv_confidence = row['Brand_IV_Prob'] / 100 * 0.5  # 50% of probability as CI width
        var_confidence = (100 - row['Brand_IV_Prob']) / 100 * 0.3  # 30% of uncertainty as CI width
        
        fig.add_trace(go.Scatter(
            x=[row['Brand_VaR10']],
            y=[row['Brand_IV']],
            mode='markers+text',
            marker=dict(
                size=row['Brand_Capture_Eff'] * 2,  # Size based on capture efficiency
                symbol=marker_symbol,
                color='lightgray',
                opacity=0.8,
                line=dict(width=2, color=marker_line_color)
            ),
            text=[row['SKU']],
            textposition="top center",
            textfont=dict(size=10, color="black"),
            name=row['SKU'],
            error_x=dict(
                type='data',
                array=[var_confidence],
                visible=True,
                width=3,
                color='rgba(0,0,0,0.3)',
                thickness=1
            ),
            error_y=dict(
                type='data',
                array=[iv_confidence],
                visible=True,
                width=3,
                color='rgba(0,0,0,0.3)',
                thickness=1
            ),
            hovertemplate=f"""
            <b>{row['SKU']}</b><br>
            Incremental Value: {row['Brand_IV']:.1f}% ±{iv_confidence:.1f}%<br>
            VaR10: {row['Brand_VaR10']:.1f}% ±{var_confidence:.1f}%<br>
            Capture Efficiency: {row['Brand_Capture_Eff']:.1f}%<br>
            Confidence: {row['Brand_IV_Prob']:.1f}%<br>
            Recommendation: {recommendation}<br>
            Composite Score: {row['Composite_Score']:.1f}
            <extra></extra>
            """
        ))
    
    # Add risk zones with standardized scales
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=4, y1=8,
        fillcolor="lightgreen",
        opacity=0.2,
        line=dict(width=0),
        name="Low Risk Zone"
    )
    
    fig.add_shape(
        type="rect",
        x0=4, y0=0, x1=6, y1=8,
        fillcolor="orange",
        opacity=0.2,
        line=dict(width=0),
        name="Medium Risk Zone"
    )
    
    fig.add_shape(
        type="rect",
        x0=6, y0=0, x1=8, y1=8,
        fillcolor="red",
        opacity=0.2,
        line=dict(width=0),
        name="High Risk Zone"
    )
    
    # Update layout with standardized scales
    fig.update_layout(
        title="Portfolio Risk Map: VaR10 vs Incremental Value",
        xaxis_title="VaR10 (%) - Risk Level",
        yaxis_title="Incremental Value (%) - Expected Return",
        height=600,
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(
            range=[0, 8],  # Standard VaR10 scale: 0-8%
            dtick=1,       # Tick every 1%
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        yaxis=dict(
            range=[0, 8],  # Standard IV scale: 0-8%
            dtick=1,       # Tick every 1%
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        annotations=[
            dict(
                x=2, y=6,
                text="Low Risk<br>High Return",
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            ),
            dict(
                x=5, y=6,
                text="Medium Risk<br>Moderate Return",
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            ),
            dict(
                x=7, y=6,
                text="High Risk<br>Variable Return",
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            ),
            dict(
                x=0.5, y=7.5,
                text="Symbols: ● Launch ● Monitor ◆ Hold",
                showarrow=False,
                font=dict(size=10, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )
    
    return fig

def get_assumption_note(chart_type):
    """
    Get appropriate assumption note for different chart types
    """
    notes = {
        'risk_map': "📊 **Note**: Risk zones based on VaR10 thresholds (≤4% Low, 4-6% Medium, >6% High). Confidence intervals show uncertainty around estimates. Data from 6 SKU portfolio analysis.",
        'source_volume': "📊 **Note**: Volume breakdown from brand vs owner perspective. Confidence intervals based on capture efficiency. Absolute values represent market share impact.",
        'mini_scatter': "📊 **Note**: Portfolio position relative to median values. Point size reflects capture efficiency. Confidence intervals show estimate uncertainty.",
        'small_multiples': "📊 **Note**: Micro-charts use standardized scales for fair comparison. Colors indicate performance quartiles. Confidence intervals shown only for key metrics to avoid clutter.",
        'category_lift': "📊 **Note**: Category lift measures potential market expansion. Actual growth may vary due to competitive response.",
        'cannibalization': "📊 **Note**: Cannibalization ratios show internal portfolio impact. Higher values indicate greater brand owner risk.",
        'composite_score': "📊 **Note**: Composite score weighted: 40% IV + 40% Capture + 20% (100-Cannibal). Higher scores indicate better launch potential."
    }
    return notes.get(chart_type, "📊 **Note**: Analysis based on current portfolio data and modeling assumptions.")

def create_what_could_change_box(row):
    """
    Create a 'what could change' box for each SKU to flag sensitivity to price or pack shifts
    """
    # Analyze sensitivity based on current metrics
    sensitivity_factors = []
    
    # Price sensitivity analysis
    if row['Brand_IV'] >= 6.0:
        sensitivity_factors.append("💰 **Price Sensitive**: High IV suggests strong price elasticity - 10% price increase could reduce demand by 15-20%")
    elif row['Brand_IV'] >= 4.0:
        sensitivity_factors.append("💰 **Moderate Price Sensitivity**: IV suggests moderate price elasticity - 10% price increase could reduce demand by 8-12%")
    else:
        sensitivity_factors.append("💰 **Low Price Sensitivity**: Lower IV suggests price stability - 10% price increase could reduce demand by 5-8%")
    
    # Pack size sensitivity
    if row['Brand_Capture_Eff'] >= 80:
        sensitivity_factors.append("📦 **Pack Size Sensitive**: High capture efficiency suggests consumers are pack-size conscious - format changes could impact 20-30% of volume")
    elif row['Brand_Capture_Eff'] >= 70:
        sensitivity_factors.append("📦 **Moderate Pack Sensitivity**: Good capture efficiency suggests some pack-size awareness - format changes could impact 10-20% of volume")
    else:
        sensitivity_factors.append("📦 **Low Pack Sensitivity**: Lower capture efficiency suggests less pack-size focus - format changes could impact 5-10% of volume")
    
    # Cannibalization sensitivity
    if row['Brand_Cannibal_Ratio'] >= 15:
        sensitivity_factors.append("⚠️ **High Cannibalization Risk**: Changes could significantly impact internal portfolio - monitor brand owner portfolio closely")
    elif row['Brand_Cannibal_Ratio'] >= 10:
        sensitivity_factors.append("⚠️ **Moderate Cannibalization Risk**: Changes could affect some internal products - watch for portfolio conflicts")
    else:
        sensitivity_factors.append("✅ **Low Cannibalization Risk**: Changes unlikely to significantly impact internal portfolio")
    
    # Category lift sensitivity
    if row['Brand_Category_Lift'] >= 6:
        sensitivity_factors.append("📈 **Category Growth Dependent**: High lift suggests reliance on category expansion - market conditions could significantly impact performance")
    elif row['Brand_Category_Lift'] >= 4:
        sensitivity_factors.append("📈 **Moderate Category Dependency**: Some lift suggests partial reliance on category growth - monitor market trends")
    else:
        sensitivity_factors.append("📈 **Low Category Dependency**: Lower lift suggests more share-based growth - less sensitive to market conditions")
    
    # Risk sensitivity
    if row['Brand_VaR10'] >= 5:
        sensitivity_factors.append("🎯 **High Risk Sensitivity**: Elevated VaR suggests high volatility - external factors could cause significant swings")
    elif row['Brand_VaR10'] >= 4:
        sensitivity_factors.append("🎯 **Moderate Risk Sensitivity**: Some VaR suggests moderate volatility - monitor external factors")
    else:
        sensitivity_factors.append("🎯 **Low Risk Sensitivity**: Lower VaR suggests stability - less sensitive to external shocks")
    
    # Create the sensitivity box
    st.markdown("**🔄 What Could Change?**")
    
    # Determine overall sensitivity level
    high_sensitivity_count = sum([
        row['Brand_IV'] >= 6.0,
        row['Brand_Capture_Eff'] >= 80,
        row['Brand_Cannibal_Ratio'] >= 15,
        row['Brand_VaR10'] >= 5
    ])
    
    if high_sensitivity_count >= 3:
        sensitivity_level = "🔴 High Sensitivity"
        sensitivity_color = "error"
    elif high_sensitivity_count >= 2:
        sensitivity_level = "🟡 Moderate Sensitivity"
        sensitivity_color = "warning"
    else:
        sensitivity_level = "🟢 Low Sensitivity"
        sensitivity_color = "success"
    
    # Display sensitivity level
    if sensitivity_color == "success":
        st.success(f"**{sensitivity_level}**")
    elif sensitivity_color == "warning":
        st.warning(f"**{sensitivity_level}**")
    else:
        st.error(f"**{sensitivity_level}**")
    
    # Display sensitivity factors
    for factor in sensitivity_factors:
        st.caption(factor)
    


def create_visual_guide_appendix():
    """
    Create an appendix page that explains how to read each visual with worked examples
    """
    st.markdown("---")
    st.markdown("## 📖 Visual Guide Appendix")
    st.markdown("*How to read and interpret each visual element in the SKU Portfolio Dashboard*")
    
    # Traffic Light System
    st.markdown("### 🚦 Traffic Light System")
    st.markdown("**Purpose**: Quick visual assessment of metric performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🟢 **GO** - Proceed with confidence")
        st.caption("**Incremental Value**: ≥6.0%")
        st.caption("**Capture Efficiency**: ≥80%")
        st.caption("**Cannibalization**: ≤10%")
        st.caption("**VaR10 Risk**: ≤4.0%")
    
    with col2:
        st.warning("🟡 **WATCH** - Monitor closely")
        st.caption("**Incremental Value**: 4.0-5.9%")
        st.caption("**Capture Efficiency**: 70-79%")
        st.caption("**Cannibalization**: 10-20%")
        st.caption("**VaR10 Risk**: 4.0-6.0%")
    
    with col3:
        st.error("🔴 **HOLD** - Needs attention")
        st.caption("**Incremental Value**: <4.0%")
        st.caption("**Capture Efficiency**: <70%")
        st.caption("**Cannibalization**: >20%")
        st.caption("**VaR10 Risk**: >6.0%")
    
    # Worked Example
    st.markdown("**📊 Worked Example**: SKU 4 shows 🟢 GO for IV (5.2%), 🟢 GO for Capture (87.0%), 🟢 GO for Cannibal (6.6%), and 🟢 GO for VaR (3.7%)")
    
    # Sparkbar System
    st.markdown("### 📊 Sparkbar Performance Indicators")
    st.markdown("**Purpose**: Show relative performance within portfolio")
    
    st.markdown("**How to Read**:")
    st.caption("• **██████████ Top 25% (Excellent)** - Performance in top quartile")
    st.caption("• **████████   Above Median (Good)** - Performance above portfolio median")
    st.caption("• **████      Below Median (Needs Review)** - Performance below median")
    
    st.markdown("**📊 Worked Example**: SKU 4's IV sparkbar shows '████████   Above Median (Good)' meaning its 5.2% IV is above the portfolio median but not in the top 25%")
    
    # Risk Map
    st.markdown("### 🗺️ Portfolio Risk Map")
    st.markdown("**Purpose**: Visualize risk-return trade-offs across all SKUs")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Axes**:")
        st.caption("• **X-axis (VaR10)**: Risk level (0-8%)")
        st.caption("• **Y-axis (Incremental Value)**: Expected return (0-8%)")
        st.caption("• **Point Size**: Capture efficiency (larger = better)")
        st.caption("• **Point Shape**: Launch recommendation")
    
    with col2:
        st.markdown("**Zones**:")
        st.caption("• **🟢 Green Zone**: Low risk, high return")
        st.caption("• **🟠 Orange Zone**: Medium risk, moderate return")
        st.caption("• **🔴 Red Zone**: High risk, variable return")
    
    st.markdown("**📊 Worked Example**: SKU 4 appears in the green zone (VaR: 3.7%, IV: 5.2%) with a large circle indicating high capture efficiency (87.0%)")
    
    # Source of Volume Chart
    st.markdown("### 📈 Source of Volume Analysis")
    st.markdown("**Purpose**: Understand where sales volume comes from")
    
    st.markdown("**Chart Elements**:")
    st.caption("• **Light Gray (Captured)**: Volume from competitors")
    st.caption("• **Dark Gray (Cannibalized)**: Volume from own portfolio")
    st.caption("• **White (Category Lift)**: Volume from category growth")
    st.caption("• **Y-axis Scale**: Standardized 0-10% for comparison")
    
    st.markdown("**📊 Worked Example**: SKU 4 shows 4.8% captured (light gray), 0.4% cannibalized (dark gray), and 0.4% category lift (white), indicating strong competitive capture with minimal internal impact")
    
    # Confidence Intervals
    st.markdown("### 📊 Confidence Intervals")
    st.markdown("**Purpose**: Show uncertainty around point estimates to help with decision-making")
    
    st.markdown("**How to Read**:")
    st.caption("• **Thin horizontal/vertical lines**: Show confidence intervals around estimates")
    st.caption("• **Shorter lines**: Higher confidence in the estimate")
    st.caption("• **Longer lines**: More uncertainty in the estimate")
    st.caption("• **Only shown on key metrics**: IV, Capture Efficiency, VaR10 to avoid clutter")
    
    st.markdown("**📊 Worked Example**: SKU 4's IV shows ±0.8% confidence interval, meaning the true IV is likely between 4.4% and 6.0%")
    
    # Small Multiples
    st.markdown("### 📊 Small Multiples Charts")
    st.markdown("**Purpose**: Compare all SKUs across key metrics")
    
    st.markdown("**How to Read**:")
    st.caption("• **Bar Height**: Metric value relative to scale")
    st.caption("• **Color Coding**: Performance quartiles")
    st.caption("  - 🟢 Green: Top 25% performance")
    st.caption("  - 🟠 Orange: Above median performance")
    st.caption("  - 🔴 Red: Below median performance")
    st.caption("• **Standardized Scales**: All charts use same scale for fair comparison")
    
    st.markdown("**📊 Worked Example**: In the IV small multiples, SKU 4 shows an orange bar at ~65% height, indicating above-median performance but not in the top quartile")
    
    # Composite Score
    st.markdown("### 🎯 Composite Score")
    st.markdown("**Purpose**: Single metric combining key performance indicators")
    
    st.markdown("**Formula**:")
    st.code("Composite Score = (IV × 0.4) + (Capture Efficiency × 0.4) + ((100 - Cannibalization) × 0.2)")
    
    st.markdown("**📊 Worked Example**: SKU 4's score = (5.2 × 0.4) + (87.0 × 0.4) + ((100-6.6) × 0.2) = 2.1 + 34.8 + 18.7 = 55.6")
    
    # Action Recommendations
    st.markdown("### 🎯 Action Recommendations")
    st.markdown("**Purpose**: Clear guidance on next steps for each SKU")
    
    st.markdown("**Recommendation Types**:")
    st.caption("• **Launch with confidence**: High IV, good capture, low cannibalization, low risk")
    st.caption("• **Pilot recommended**: Moderate metrics, some risk factors")
    st.caption("• **Defer launch**: Low IV, high risk, or high cannibalization")
    
    st.markdown("**Confidence Levels**:")
    st.caption("• **High (80-100)**: Strong evidence for recommendation")
    st.caption("• **Medium (60-79)**: Good evidence with some uncertainty")
    st.caption("• **Low (40-59)**: Limited evidence, consider additional analysis")
    
    st.markdown("**📊 Worked Example**: SKU 4 gets 'Launch with confidence' at High confidence (95/100) based on strong metrics across all dimensions")
    
    # Key Insights
    st.markdown("### 💡 How to Use This Dashboard")
    st.markdown("**Recommended Workflow**:")
    st.caption("1. **Start with Overview Tab** - Get big picture of all SKUs with dual perspectives")
    st.caption("2. **Use Filters** - Focus on specific performance tiers, risk levels, or launch readiness")
    st.caption("3. **Check Risk Map** - Identify high-potential, low-risk opportunities")
    st.caption("4. **Review Cannibalization Watch** - Monitor internal portfolio impact")
    st.caption("5. **Check Category Lift** - Understand true expansion potential")
    st.caption("6. **Use Small Multiples** - Compare SKUs side-by-side")
    st.caption("7. **Check Decision Tab** - Get clear Launch/Pilot/Defer recommendations")
    
    # Filter Behavior Note
    st.markdown("### ⚠️ Filter Behavior Note")
    st.markdown("**Important**: When you use any filter for the first time, the dashboard will jump to the first tab. After that, changing filter values will keep you on your current tab. This is a platform limitation and doesn't affect the data or functionality.")
    
    st.markdown("**📊 Remember**: All charts use standardized scales and color-sparing design for print accessibility and reliable visual comparisons.")
    
    # Current Dashboard Features
    st.markdown("---")
    st.markdown("### 🆕 Current Dashboard Features")
    st.markdown("*Key features and recent updates to the dashboard*")
    
    # SKU Level Analysis
    st.markdown("#### 📊 SKU Level Analysis")
    st.markdown("**Purpose**: Shows both Brand and Owner perspectives simultaneously for each SKU")
    
    st.markdown("**How it Works**:")
    st.caption("• **Dual Perspective**: Each metric shows both Brand and Owner values")
    st.caption("• **Format**: '📊 Brand: X% | 🏢 Owner: Y%'")
    st.caption("• **Location**: All SKU detail views in Overview tab")
    st.caption("• **Benefit**: Complete picture without switching perspectives")
    
    st.markdown("**📊 Worked Example**: IV shows '📊 Brand: 4.8% | 🏢 Owner: 3.4%' - you see both perspectives at once")
    
    # Bookmarking System
    st.markdown("#### ⭐ Bookmarking System")
    st.markdown("**Purpose**: Mark SKUs for follow-up and executive review")
    
    st.markdown("**How to Use**:")
    st.caption("• **Star Icon**: Click ☆ to bookmark, ⭐ to remove")
    st.caption("• **Location**: Top-right of each SKU card")
    st.caption("• **Sidebar Panel**: View all bookmarked SKUs")
    st.caption("• **Clear All**: Remove all bookmarks at once")
    
    st.markdown("**📊 Worked Example**: Bookmark SKU 4 by clicking the star icon, then check the sidebar to see it in your 'Bookmarked SKUs' list")
    
    # Simple Decision Page
    st.markdown("#### 🎯 Simple Decision Page")
    st.markdown("**Purpose**: Clear Launch/Pilot/Defer recommendations for executive decision-making")
    
    st.markdown("**How to Read**:")
    st.caption("• **🚀 Launch Now**: High performance, low risk - ready for immediate launch")
    st.caption("• **🧪 Pilot**: Moderate performance - test in controlled environment first")
    st.caption("• **⏸️ Defer**: Low performance or high risk - needs optimization")
    
    st.markdown("**Decision Criteria**:")
    st.caption("• **Launch**: IV≥5.0%, Capture≥70.0%, Cannibal≤15.0%, VaR≤5.0%, Confidence≥80/100")
    st.caption("• **Pilot**: IV≥4.0%, Capture≥65.0%, Cannibal≤20.0%, VaR≤6.0%, Confidence 60-79/100")
    st.caption("• **Defer**: Any metric below pilot thresholds")
    
    st.markdown("**📊 Worked Example**: SKU 4 appears in 'Launch Now' category with score 55.6 and High confidence")
    
    # Available Tabs
    st.markdown("#### 📋 Available Tabs")
    st.markdown("**Current Dashboard Structure**:")
    st.caption("• **How to use the dashboard** - This guide")
    st.caption("• **📈 Overview** - Portfolio leaderboard with dual perspectives")
    st.caption("• **🗺️ Risk** - Portfolio risk map and analysis")
    st.caption("• **⚠️ Cannibal** - Cannibalization watchlist")
    st.caption("• **📈 Lift** - Category lift analysis")
    st.caption("• **📊 Multiples** - Small multiples comparison")
    st.caption("• **🎯 Rivals** - Competitive analysis")
    st.caption("• **📊 Compact** - Space-efficient view")
    st.caption("• **📋 Executive** - Executive booklet (3 pages)")
    st.caption("• **🔍 Details** - Complete data view")
    st.caption("• **📋 Reference** - Definitions and thresholds")
    st.caption("• **🎯 Decision** - Launch/Pilot/Defer recommendations")

def create_confidence_meter(probability_value):
    """
    Create a confidence meter visual mapping Probability IV>0 to clear labels
    """
    # Define confidence levels and their ranges
    if probability_value >= 95:
        confidence_level = "Very High"
        confidence_color = "success"
        confidence_emoji = "🟢"
        confidence_description = "Extremely confident in positive IV"
    elif probability_value >= 90:
        confidence_level = "High"
        confidence_color = "success"
        confidence_emoji = "🟢"
        confidence_description = "Very confident in positive IV"
    elif probability_value >= 85:
        confidence_level = "Good"
        confidence_color = "success"
        confidence_emoji = "🟢"
        confidence_description = "Good confidence in positive IV"
    elif probability_value >= 80:
        confidence_level = "Moderate"
        confidence_color = "warning"
        confidence_emoji = "🟡"
        confidence_description = "Moderate confidence in positive IV"
    elif probability_value >= 70:
        confidence_level = "Low"
        confidence_color = "warning"
        confidence_emoji = "🟡"
        confidence_description = "Low confidence in positive IV"
    else:
        confidence_level = "Very Low"
        confidence_color = "error"
        confidence_emoji = "🔴"
        confidence_description = "Very low confidence in positive IV"
    
    # Create visual meter using progress bar
    progress_value = probability_value / 100.0
    
    # Create confidence meter display
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown(f"**{confidence_emoji} {confidence_level}**")
    
    with col2:
        if confidence_color == "success":
            st.progress(progress_value)
        elif confidence_color == "warning":
            st.progress(progress_value)
        else:
            st.progress(progress_value)
    
    with col3:
        st.markdown(f"**{probability_value:.1f}%**")
    
    # Add description
    if confidence_color == "success":
        st.success(f"📊 {confidence_description}")
    elif confidence_color == "warning":
        st.warning(f"📊 {confidence_description}")
    else:
        st.error(f"📊 {confidence_description}")
    
    return confidence_level, confidence_color, confidence_description

def create_executive_booklet(df):
    """
    Create an executive booklet: 3 pages total (leaderboard, risk map, Top 5), with links to detail pages
    """
    st.markdown("### 📋 Executive Booklet")
    st.markdown("*3-page executive summary with links to detailed analysis*")
    
    # Page selector
    page = st.radio(
        "📄 Select Page:",
        ["📊 Page 1: Portfolio Leaderboard", "🗺️ Page 2: Risk Map", "🏆 Page 3: Top 5 Launch Ready"],
        horizontal=True
    )
    
    if "Page 1" in page:
        create_executive_page1_leaderboard(df)
    elif "Page 2" in page:
        create_executive_page2_risk_map(df)
    elif "Page 3" in page:
        create_executive_page3_top5(df)

def get_ordinal_suffix(num):
    """Convert number to ordinal (1st, 2nd, 3rd, 4th, etc.)"""
    if 10 <= num % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(num % 10, 'th')
    return f"{num}{suffix}"

def create_sku_ranking_footer(sku_row, df):
    """
    Create a page footer showing where this SKU ranks on each metric within the portfolio
    """
    st.markdown("---")
    st.markdown("### 📊 Portfolio Ranking Summary")
    st.markdown(f"*Where {sku_row['SKU']} ranks across all metrics in the portfolio*")
    
    # Calculate rankings for each metric based on current perspective
    rankings = {}
    perspective = st.session_state.get('perspective', 'Brand')
    
    # Get metric column names based on perspective
    iv_col = 'Brand_IV' if perspective == 'Brand' else 'Owner_IV'
    capture_col = 'Brand_Capture_Eff' if perspective == 'Brand' else 'Owner_Capture_Eff'
    cannibal_col = 'Brand_Cannibal_Ratio' if perspective == 'Brand' else 'Owner_Cannibal_Ratio'
    var_col = 'Brand_VaR10' if perspective == 'Brand' else 'Owner_VaR10'
    lift_col = 'Brand_Category_Lift' if perspective == 'Brand' else 'Owner_Category_Lift'
    prob_col = 'Brand_IV_Prob' if perspective == 'Brand' else 'Owner_IV_Prob'
    
    # IV ranking (higher is better)
    iv_rank = (df[iv_col] > sku_row[iv_col]).sum() + 1
    iv_percentile = (len(df) - iv_rank + 1) / len(df) * 100
    rankings['IV'] = {'rank': iv_rank, 'percentile': iv_percentile, 'better': 'higher', 'value': sku_row[iv_col]}
    
    # Capture Efficiency ranking (higher is better)
    capture_rank = (df[capture_col] > sku_row[capture_col]).sum() + 1
    capture_percentile = (len(df) - capture_rank + 1) / len(df) * 100
    rankings['Capture_Eff'] = {'rank': capture_rank, 'percentile': capture_percentile, 'better': 'higher', 'value': sku_row[capture_col]}
    
    # Cannibalization ranking (lower is better)
    cannibal_rank = (df[cannibal_col] < sku_row[cannibal_col]).sum() + 1
    cannibal_percentile = cannibal_rank / len(df) * 100
    rankings['Cannibal_Ratio'] = {'rank': cannibal_rank, 'percentile': cannibal_percentile, 'better': 'lower', 'value': sku_row[cannibal_col]}
    
    # VaR10 ranking (lower is better)
    var_rank = (df[var_col] < sku_row[var_col]).sum() + 1
    var_percentile = var_rank / len(df) * 100
    rankings['VaR10'] = {'rank': var_rank, 'percentile': var_percentile, 'better': 'lower', 'value': sku_row[var_col]}
    
    # Category Lift ranking (higher is better)
    lift_rank = (df[lift_col] > sku_row[lift_col]).sum() + 1
    lift_percentile = (len(df) - lift_rank + 1) / len(df) * 100
    rankings['Category_Lift'] = {'rank': lift_rank, 'percentile': lift_percentile, 'better': 'higher', 'value': sku_row[lift_col]}
    
    # Market HHI ranking (lower is better)
    hhi_rank = (df['Market_HHI'] < sku_row['Market_HHI']).sum() + 1
    hhi_percentile = hhi_rank / len(df) * 100
    rankings['Market_HHI'] = {'rank': hhi_rank, 'percentile': hhi_percentile, 'better': 'lower', 'value': sku_row['Market_HHI']}
    
    # Probability IV>0 ranking (higher is better)
    prob_rank = (df[prob_col] > sku_row[prob_col]).sum() + 1
    prob_percentile = (len(df) - prob_rank + 1) / len(df) * 100
    rankings['IV_Prob'] = {'rank': prob_rank, 'percentile': prob_percentile, 'better': 'higher', 'value': sku_row[prob_col]}
    
    # Composite Score ranking (higher is better)
    score_rank = (df['Composite_Score'] > sku_row['Composite_Score']).sum() + 1
    score_percentile = (len(df) - score_rank + 1) / len(df) * 100
    rankings['Composite_Score'] = {'rank': score_rank, 'percentile': score_percentile, 'better': 'higher', 'value': sku_row['Composite_Score']}
    
    # Display rankings in organized columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**💰 Value Metrics**")
        
        # IV
        rank_data = rankings['IV']
        if rank_data['percentile'] >= 80:
            rank_emoji = "🥇"
            rank_color = "success"
        elif rank_data['percentile'] >= 60:
            rank_emoji = "🥈"
            rank_color = "info"
        elif rank_data['percentile'] >= 40:
            rank_emoji = "🥉"
            rank_color = "warning"
        else:
            rank_emoji = "📉"
            rank_color = "error"
        
        st.markdown(f"**Incremental Value**: {rank_emoji} Rank #{rank_data['rank']} ({get_ordinal_suffix(int(rank_data['percentile']))} percentile)")
        st.caption(f"Value: {rank_data['value']:.1f}% | {rank_data['better'].title()} is better")
        
        # Composite Score
        rank_data = rankings['Composite_Score']
        if rank_data['percentile'] >= 80:
            rank_emoji = "🥇"
            rank_color = "success"
        elif rank_data['percentile'] >= 60:
            rank_emoji = "🥈"
            rank_color = "info"
        elif rank_data['percentile'] >= 40:
            rank_emoji = "🥉"
            rank_color = "warning"
        else:
            rank_emoji = "📉"
            rank_color = "error"
        
        st.markdown(f"**Composite Score**: {rank_emoji} Rank #{rank_data['rank']} ({get_ordinal_suffix(int(rank_data['percentile']))} percentile)")
        st.caption(f"Score: {rank_data['value']:.1f} | {rank_data['better'].title()} is better")
    
    with col2:
        st.markdown("**🎯 Performance Metrics**")
        
        # Capture Efficiency
        rank_data = rankings['Capture_Eff']
        if rank_data['percentile'] >= 80:
            rank_emoji = "🥇"
        elif rank_data['percentile'] >= 60:
            rank_emoji = "🥈"
        elif rank_data['percentile'] >= 40:
            rank_emoji = "🥉"
        else:
            rank_emoji = "📉"
        
        st.markdown(f"**Capture Efficiency**: {rank_emoji} Rank #{rank_data['rank']} ({get_ordinal_suffix(int(rank_data['percentile']))} percentile)")
        st.caption(f"Efficiency: {rank_data['value']:.1f}% | {rank_data['better'].title()} is better")
        
        # Category Lift
        rank_data = rankings['Category_Lift']
        if rank_data['percentile'] >= 80:
            rank_emoji = "🥇"
        elif rank_data['percentile'] >= 60:
            rank_emoji = "🥈"
        elif rank_data['percentile'] >= 40:
            rank_emoji = "🥉"
        else:
            rank_emoji = "📉"
        
        st.markdown(f"**Category Lift**: {rank_emoji} Rank #{rank_data['rank']} ({get_ordinal_suffix(int(rank_data['percentile']))} percentile)")
        st.caption(f"Lift: {rank_data['value']:.1f}% | {rank_data['better'].title()} is better")
    
    with col3:
        st.markdown("**⚠️ Risk Metrics**")
        
        # Cannibalization
        rank_data = rankings['Cannibal_Ratio']
        if rank_data['percentile'] >= 80:
            rank_emoji = "🥇"
        elif rank_data['percentile'] >= 60:
            rank_emoji = "🥈"
        elif rank_data['percentile'] >= 40:
            rank_emoji = "🥉"
        else:
            rank_emoji = "📉"
        
        st.markdown(f"**Cannibalization**: {rank_emoji} Rank #{rank_data['rank']} ({get_ordinal_suffix(int(rank_data['percentile']))} percentile)")
        st.caption(f"Ratio: {rank_data['value']:.1f}% | {rank_data['better'].title()} is better")
        
        # VaR10
        rank_data = rankings['VaR10']
        if rank_data['percentile'] >= 80:
            rank_emoji = "🥇"
        elif rank_data['percentile'] >= 60:
            rank_emoji = "🥈"
        elif rank_data['percentile'] >= 40:
            rank_emoji = "🥉"
        else:
            rank_emoji = "📉"
        
        st.markdown(f"**VaR10**: {rank_emoji} Rank #{rank_data['rank']} ({get_ordinal_suffix(int(rank_data['percentile']))} percentile)")
        st.caption(f"VaR: {rank_data['value']:.1f}% | {rank_data['better'].title()} is better")
    
    with col4:
        st.markdown("**📊 Confidence & Market**")
        
        # Probability IV>0
        rank_data = rankings['IV_Prob']
        if rank_data['percentile'] >= 80:
            rank_emoji = "🥇"
        elif rank_data['percentile'] >= 60:
            rank_emoji = "🥈"
        elif rank_data['percentile'] >= 40:
            rank_emoji = "🥉"
        else:
            rank_emoji = "📉"
        
        st.markdown(f"**Confidence**: {rank_emoji} Rank #{rank_data['rank']} ({get_ordinal_suffix(int(rank_data['percentile']))} percentile)")
        st.caption(f"Probability: {rank_data['value']:.1f}% | {rank_data['better'].title()} is better")
        
        # Market HHI
        rank_data = rankings['Market_HHI']
        if rank_data['percentile'] >= 80:
            rank_emoji = "🥇"
        elif rank_data['percentile'] >= 60:
            rank_emoji = "🥈"
        elif rank_data['percentile'] >= 40:
            rank_emoji = "🥉"
        else:
            rank_emoji = "📉"
        
        st.markdown(f"**Market HHI**: {rank_emoji} Rank #{rank_data['rank']} ({get_ordinal_suffix(int(rank_data['percentile']))} percentile)")
        st.caption(f"HHI: {rank_data['value']:.2f}% | {rank_data['better'].title()} is better")
    
    # Overall portfolio position summary
    st.markdown("---")
    st.markdown("#### 🏆 Overall Portfolio Position")
    
    # Calculate overall performance score based on current perspective
    performance_metrics = ['IV', 'Capture_Eff', 'Category_Lift', 'IV_Prob', 'Composite_Score']
    risk_metrics = ['Cannibal_Ratio', 'VaR10', 'Market_HHI']
    
    avg_performance_percentile = sum(rankings[metric]['percentile'] for metric in performance_metrics) / len(performance_metrics)
    avg_risk_percentile = sum(rankings[metric]['percentile'] for metric in risk_metrics) / len(risk_metrics)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if avg_performance_percentile >= 80:
            st.success(f"🎯 **Performance**: Top Tier ({get_ordinal_suffix(int(avg_performance_percentile))} percentile)")
        elif avg_performance_percentile >= 60:
            st.info(f"📈 **Performance**: Above Average ({get_ordinal_suffix(int(avg_performance_percentile))} percentile)")
        elif avg_performance_percentile >= 40:
            st.warning(f"📊 **Performance**: Average ({get_ordinal_suffix(int(avg_performance_percentile))} percentile)")
        else:
            st.error(f"📉 **Performance**: Below Average ({get_ordinal_suffix(int(avg_performance_percentile))} percentile)")
    
    with col2:
        if avg_risk_percentile >= 80:
            st.success(f"🛡️ **Risk Profile**: Low Risk ({get_ordinal_suffix(int(avg_risk_percentile))} percentile)")
        elif avg_risk_percentile >= 60:
            st.info(f"⚖️ **Risk Profile**: Moderate Risk ({get_ordinal_suffix(int(avg_risk_percentile))} percentile)")
        elif avg_risk_percentile >= 40:
            st.warning(f"⚠️ **Risk Profile**: Higher Risk ({get_ordinal_suffix(int(avg_risk_percentile))} percentile)")
        else:
            st.error(f"🚨 **Risk Profile**: High Risk ({get_ordinal_suffix(int(avg_risk_percentile))} percentile)")
    
    with col3:
        # Calculate overall rank
        overall_rank = (df['Composite_Score'] > sku_row['Composite_Score']).sum() + 1
        overall_percentile = (len(df) - overall_rank + 1) / len(df) * 100
        
        if overall_percentile >= 80:
            st.success(f"🏆 **Overall Rank**: #{overall_rank} of {len(df)} (Top {100-overall_percentile:.0f}%)")
        elif overall_percentile >= 60:
            st.info(f"📊 **Overall Rank**: #{overall_rank} of {len(df)} (Above Average)")
        elif overall_percentile >= 40:
            st.warning(f"📈 **Overall Rank**: #{overall_rank} of {len(df)} (Average)")
        else:
            st.error(f"📉 **Overall Rank**: #{overall_rank} of {len(df)} (Below Average)")
    
    st.caption("📝 **Note**: Rankings based on current portfolio of SKUs. Percentiles show relative performance within the portfolio.")

def get_metric_by_perspective(metric_base_name, row):
    """
    Get the correct metric value based on current perspective (Brand vs Owner)
    """
    perspective = st.session_state.get('perspective', 'Brand')
    
    metric_mapping = {
        'IV': 'Brand_IV' if perspective == 'Brand' else 'Owner_IV',
        'IV_Prob': 'Brand_IV_Prob' if perspective == 'Brand' else 'Owner_IV_Prob',
        'VaR10': 'Brand_VaR10' if perspective == 'Brand' else 'Owner_VaR10',
        'Capture_Eff': 'Brand_Capture_Eff' if perspective == 'Brand' else 'Owner_Capture_Eff',
        'Cannibal_Ratio': 'Brand_Cannibal_Ratio' if perspective == 'Brand' else 'Owner_Cannibal_Ratio',
        'Category_Lift': 'Brand_Category_Lift' if perspective == 'Brand' else 'Owner_Category_Lift'
    }
    
    if metric_base_name in metric_mapping:
        return row[metric_mapping[metric_base_name]]
    else:
        return row.get(metric_base_name, 0)

def get_metric_name_by_perspective(metric_base_name):
    """
    Get the correct metric column name based on current perspective
    """
    perspective = st.session_state.get('perspective', 'Brand')
    
    metric_mapping = {
        'IV': 'Brand_IV' if perspective == 'Brand' else 'Owner_IV',
        'IV_Prob': 'Brand_IV_Prob' if perspective == 'Brand' else 'Owner_IV_Prob',
        'VaR10': 'Brand_VaR10' if perspective == 'Brand' else 'Owner_VaR10',
        'Capture_Eff': 'Brand_Capture_Eff' if perspective == 'Brand' else 'Owner_Capture_Eff',
        'Cannibal_Ratio': 'Brand_Cannibal_Ratio' if perspective == 'Brand' else 'Owner_Cannibal_Ratio',
        'Category_Lift': 'Brand_Category_Lift' if perspective == 'Brand' else 'Owner_Category_Lift'
    }
    
    return metric_mapping.get(metric_base_name, metric_base_name)

def get_perspective_label():
    """
    Get the current perspective label for display
    """
    perspective = st.session_state.get('perspective', 'Brand')
    return f"({perspective} View)"

def get_dual_metric_display(metric_base_name, row):
    """
    Get both Brand and Owner metric values for dual perspective display
    """
    metric_mapping = {
        'IV': ('Brand_IV', 'Owner_IV'),
        'IV_Prob': ('Brand_IV_Prob', 'Owner_IV_Prob'),
        'VaR10': ('Brand_VaR10', 'Owner_VaR10'),
        'Capture_Eff': ('Brand_Capture_Eff', 'Owner_Capture_Eff'),
        'Cannibal_Ratio': ('Brand_Cannibal_Ratio', 'Owner_Cannibal_Ratio'),
        'Category_Lift': ('Brand_Category_Lift', 'Owner_Category_Lift')
    }
    
    if metric_base_name in metric_mapping:
        brand_col, owner_col = metric_mapping[metric_base_name]
        return row[brand_col], row[owner_col]
    else:
        return row.get(metric_base_name, 0), row.get(metric_base_name, 0)

def create_dual_metric_display(metric_base_name, row, max_value=None, reverse=False):
    """
    Create a dual perspective display showing both Brand and Owner metrics
    """
    brand_value, owner_value = get_dual_metric_display(metric_base_name, row)
    
    # Create display with both values
    display_text = f"📊 **Brand**: {brand_value:.1f}% | 🏢 **Owner**: {owner_value:.1f}%"
    
    # Add visual bars for both
    if max_value:
        brand_bar = create_in_cell_bar(brand_value, max_value, 8, reverse)
        owner_bar = create_in_cell_bar(owner_value, max_value, 8, reverse)
        display_text += f"\n📊 Brand: {brand_bar}\n🏢 Owner: {owner_bar}"
    
    return display_text

def get_plain_english_headline(metric_name, value, context=""):
    """
    Convert technical metrics to plain-english headlines that C-level can understand
    """
    # Normalize metric name to base name for both Brand and Owner perspectives
    base_metric = metric_name.replace('Brand_', '').replace('Owner_', '')
    
    headlines = {
        'Capture_Eff': {
            'high': lambda v: f"Gains mostly from competitors ({v:.1f}%)",
            'medium': lambda v: f"Mixed gains from competitors and category ({v:.1f}%)",
            'low': lambda v: f"Limited competitive gains ({v:.1f}%)"
        },
        'Cannibal_Ratio': {
            'high': lambda v: f"Watch internal cannibalization ({v:.1f}%)",
            'medium': lambda v: f"Some internal impact expected ({v:.1f}%)",
            'low': lambda v: f"Minimal internal impact ({v:.1f}%)"
        },
        'VaR10': {
            'high': lambda v: f"High risk profile ({v:.1f}%)",
            'medium': lambda v: f"Moderate risk profile ({v:.1f}%)",
            'low': lambda v: f"Low risk profile ({v:.1f}%)"
        },
        'Category_Lift': {
            'high': lambda v: f"Expands the category significantly ({v:.1f}%)",
            'medium': lambda v: f"Moderate category expansion ({v:.1f}%)",
            'low': lambda v: f"Limited category impact ({v:.1f}%)"
        },
        'IV': {
            'high': lambda v: f"Strong incremental value ({v:.1f}%)",
            'medium': lambda v: f"Moderate incremental value ({v:.1f}%)",
            'low': lambda v: f"Limited incremental value ({v:.1f}%)"
        },
        'IV_Prob': {
            'high': lambda v: f"High confidence in success ({v:.1f}%)",
            'medium': lambda v: f"Moderate confidence ({v:.1f}%)",
            'low': lambda v: f"Low confidence in success ({v:.1f}%)"
        },
        'Market_HHI': {
            'high': lambda v: f"Concentrated market ({v:.2f}%)",
            'medium': lambda v: f"Moderately competitive market ({v:.2f}%)",
            'low': lambda v: f"Highly competitive market ({v:.2f}%)"
        }
    }
    
    # Define thresholds for each metric (same for Brand and Owner)
    thresholds = {
        'Capture_Eff': {'high': 75, 'medium': 50},
        'Cannibal_Ratio': {'high': 15, 'medium': 8},
        'VaR10': {'high': 6, 'medium': 4},
        'Category_Lift': {'high': 6, 'medium': 4},
        'IV': {'high': 5, 'medium': 3},
        'IV_Prob': {'high': 90, 'medium': 80},
        'Market_HHI': {'high': 8, 'medium': 5}
    }
    
    if base_metric in headlines and base_metric in thresholds:
        if value >= thresholds[base_metric]['high']:
            level = 'high'
        elif value >= thresholds[base_metric]['medium']:
            level = 'medium'
        else:
            level = 'low'
        
        return headlines[base_metric][level](value)
    
    # Fallback for unknown metrics
    return f"{metric_name}: {value:.1f}%"

def get_plain_english_summary(sku_row):
    """
    Generate a plain-english summary for the entire SKU
    """
    # Key insights
    insights = []
    
    # Capture efficiency insight
    if sku_row['Brand_Capture_Eff'] >= 75:
        insights.append("Strong competitive positioning")
    elif sku_row['Brand_Capture_Eff'] >= 50:
        insights.append("Moderate competitive gains")
    else:
        insights.append("Limited competitive advantage")
    
    # Cannibalization insight
    if sku_row['Brand_Cannibal_Ratio'] <= 8:
        insights.append("minimal internal impact")
    elif sku_row['Brand_Cannibal_Ratio'] <= 15:
        insights.append("some internal trade-offs")
    else:
        insights.append("significant internal cannibalization")
    
    # Risk insight
    if sku_row['Brand_VaR10'] <= 4:
        insights.append("low risk profile")
    elif sku_row['Brand_VaR10'] <= 6:
        insights.append("moderate risk")
    else:
        insights.append("high risk profile")
    
    # Category lift insight
    if sku_row['Brand_Category_Lift'] >= 6:
        insights.append("expands the category")
    elif sku_row['Brand_Category_Lift'] >= 4:
        insights.append("moderate category growth")
    else:
        insights.append("limited category impact")
    
    # Confidence insight
    if sku_row['Brand_IV_Prob'] >= 90:
        insights.append("high confidence in success")
    elif sku_row['Brand_IV_Prob'] >= 80:
        insights.append("good confidence")
    else:
        insights.append("uncertain outcomes")
    
    # Combine insights
    if len(insights) >= 3:
        summary = f"This SKU shows {insights[0]} with {insights[1]} and a {insights[2]}. "
        if len(insights) >= 4:
            summary += f"It {insights[3]} with {insights[4]}."
    else:
        summary = f"Performance shows {', '.join(insights)}."
    
    return summary

def get_plain_english_action(sku_row):
    """
    Generate plain-english action recommendations
    """
    recommendation, confidence, confidence_score, rec_color = get_action_recommendation(sku_row)
    
    # Convert technical recommendations to plain English
    if "Launch with high confidence" in recommendation:
        return "🚀 **Ready to launch** - Strong performance indicators suggest immediate market entry"
    elif "Launch with moderate confidence" in recommendation:
        return "🚀 **Launch recommended** - Good performance with manageable risks"
    elif "Pilot test recommended" in recommendation:
        return "🧪 **Pilot test first** - Validate performance in controlled environment"
    elif "Defer launch" in recommendation:
        return "⏸️ **Defer launch** - Address performance concerns before market entry"
    else:
        return f"📊 **Action needed** - {recommendation}"

def create_executive_page1_leaderboard(df):
    """
    Executive Page 1: Portfolio Leaderboard
    """
    # Check if DataFrame is empty
    if df.empty:
        st.warning("⚠️ No SKUs available for analysis. Please adjust your filters.")
        return
    
    st.markdown("---")
    st.markdown("## 📊 EXECUTIVE SUMMARY - PORTFOLIO LEADERBOARD")
    st.markdown("*Page 1 of 3 - SKU Performance Ranking*")
    
    # Executive summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total SKUs", len(df))
    with col2:
        avg_score = df['Composite_Score'].mean()
        st.metric("Avg Score", f"{avg_score:.1f}")
    with col3:
        high_performers = len(df[df['Composite_Score'] >= 80])
        st.metric("High Performers", high_performers)
    with col4:
        launch_ready = len(df[(df['Brand_IV'] >= 5) & (df['Brand_Capture_Eff'] >= 75)])
        st.metric("Launch Ready", launch_ready)
    
    st.markdown("---")
    
    # Top 10 SKUs with executive-level detail
    st.markdown("### 🏆 TOP 10 SKU PERFORMANCE")
    
    top_10 = df.head(10)
    
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        recommendation, confidence, _, rec_color = get_action_recommendation(row)
        
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            st.markdown(f"**#{idx}**")
        
        with col2:
            st.markdown(f"**{row['SKU']}**")
            st.caption(f"Score: {row['Composite_Score']:.1f}")
        
        with col3:
                        # Key metrics in compact format
                        st.caption(f"💰 IV: {row['Brand_IV']:.1f}% | 🎯 Capture: {row['Brand_Capture_Eff']:.1f}%")
                        st.caption(f"♻️ Cannibal: {row['Brand_Cannibal_Ratio']:.1f}% | 🛡️ VaR: {row['Brand_VaR10']:.1f}%")
        
        with col4:
            if rec_color == "success":
                st.success(confidence)
            elif rec_color == "warning":
                st.warning(confidence)
            else:
                st.error(confidence)
        
        st.markdown("---")
    
    # Quick insights
    st.markdown("### 💡 KEY INSIGHTS")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Performance Highlights**")
        best_iv = df.loc[df['Brand_IV'].idxmax()]
        st.caption(f"• Highest IV: {best_iv['SKU']} ({best_iv['Brand_IV']:.1f}%)")
        
        best_capture = df.loc[df['Brand_Capture_Eff'].idxmax()]
        st.caption(f"• Best Capture: {best_capture['SKU']} ({best_capture['Brand_Capture_Eff']:.1f}%)")
        
        lowest_cannibal = df.loc[df['Brand_Cannibal_Ratio'].idxmin()]
        st.caption(f"• Lowest Cannibal: {lowest_cannibal['SKU']} ({lowest_cannibal['Brand_Cannibal_Ratio']:.1f}%)")
    
    with col2:
        st.markdown("**📊 Portfolio Health**")
        st.caption(f"• {len(df[df['Composite_Score'] >= 80])} high-performing SKUs")
        st.caption(f"• {len(df[df['Brand_IV'] >= 5])} SKUs with strong IV")
        st.caption(f"• {len(df[df['Brand_Cannibal_Ratio'] <= 10])} SKUs with low cannibalization")
    
    # Navigation
    st.markdown("---")

def create_executive_page2_risk_map(df):
    """
    Executive Page 2: Risk Map
    """
    # Check if DataFrame is empty
    if df.empty:
        st.warning("⚠️ No SKUs available for analysis. Please adjust your filters.")
        return
    
    st.markdown("---")
    st.markdown("## 🗺️ EXECUTIVE SUMMARY - RISK ANALYSIS")
    st.markdown("*Page 2 of 3 - Portfolio Risk Assessment*")
    
    # Risk summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    low_risk = len(df[(df['Brand_VaR10'] <= 4) & (df['Brand_IV'] >= 5)])
    medium_risk = len(df[(df['Brand_VaR10'] > 4) & (df['Brand_VaR10'] <= 6)])
    high_risk = len(df[df['Brand_VaR10'] > 6])
    
    with col1:
        st.metric("🟢 Low Risk", low_risk)
    with col2:
        st.metric("🟡 Medium Risk", medium_risk)
    with col3:
        st.metric("🔴 High Risk", high_risk)
    with col4:
        avg_var = df['Brand_VaR10'].mean()
        st.metric("Avg VaR", f"{avg_var:.1f}%")
    
    st.markdown("---")
    
    # Risk map visualization
    st.markdown("### 🗺️ PORTFOLIO RISK MAP")
    
    # Create simplified risk map
    fig = go.Figure()
    
    for idx, row in df.iterrows():
        recommendation, confidence, _, rec_color = get_action_recommendation(row)
        
        # Determine marker style based on risk level
        if row['Brand_VaR10'] <= 4:
            symbol = 'circle'
            color = 'green'
        elif row['Brand_VaR10'] <= 6:
            symbol = 'square'
            color = 'orange'
        else:
            symbol = 'diamond'
            color = 'red'
        
        fig.add_trace(go.Scatter(
            x=[row['Brand_IV']],
            y=[row['Brand_VaR10']],
            mode='markers+text',
            marker=dict(
                size=15,
                color=color,
                symbol=symbol,
                line=dict(width=2, color='black')
            ),
            text=[row['SKU']],
            textposition="top center",
            textfont=dict(color='black', size=10),
            name=row['SKU'],
            hovertemplate=f"<b>{row['SKU']}</b><br>" +
                         f"IV: {row['Brand_IV']:.1f}%<br>" +
                         f"VaR: {row['Brand_VaR10']:.1f}%<br>" +
                         f"Risk: {'Low' if row['Brand_VaR10'] <= 4 else 'Medium' if row['Brand_VaR10'] <= 6 else 'High'}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Portfolio Risk Map: IV vs VaR",
        xaxis_title="Incremental Value (%)",
        yaxis_title="VaR10 (%)",
        xaxis=dict(range=[0, 8]),
        yaxis=dict(range=[0, 8]),
        showlegend=False,
        height=500,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk zones explanation
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🟢 **Green Zone**: Low Risk, High Return - Launch with confidence")
    with col2:
        st.warning("🟠 **Orange Zone**: Medium Risk - Monitor closely")
    with col3:
        st.error("🔴 **Red Zone**: High Risk - Consider pilot testing")
    
    # Risk insights
    st.markdown("### 💡 RISK INSIGHTS")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Risk Distribution**")
        st.caption(f"• {low_risk/len(df)*100:.0f}% of SKUs are low risk")
        st.caption(f"• {medium_risk/len(df)*100:.0f}% of SKUs are medium risk")
        st.caption(f"• {high_risk/len(df)*100:.0f}% of SKUs are high risk")
    
    with col2:
        st.markdown("**⚠️ Risk Alerts**")
        high_risk_skus = df[df['Brand_VaR10'] > 6]
        if len(high_risk_skus) > 0:
            st.caption("• High risk SKUs requiring attention:")
            for _, sku in high_risk_skus.head(3).iterrows():
                st.caption(f"  - {sku['SKU']} (VaR: {sku['Brand_VaR10']:.1f}%)")
        else:
            st.caption("• No high risk SKUs identified")
    
    # Navigation
    st.markdown("---")

def create_executive_page3_top5(df):
    """
    Executive Page 3: Top 5 Launch Ready
    """
    # Check if DataFrame is empty
    if df.empty:
        st.warning("⚠️ No SKUs available for analysis. Please adjust your filters.")
        return
    
    st.markdown("---")
    st.markdown("## 🏆 EXECUTIVE SUMMARY - TOP 5 LAUNCH READY")
    st.markdown("*Page 3 of 3 - Immediate Launch Recommendations*")
    
    # Launch readiness summary
    col1, col2, col3, col4 = st.columns(4)
    
    launch_ready = df[(df['Brand_IV'] >= 5) & (df['Brand_Capture_Eff'] >= 75) & (df['Brand_Cannibal_Ratio'] <= 15)]
    pilot_ready = df[(df['Brand_IV'] >= 3) & (df['Brand_Capture_Eff'] >= 60)]
    defer = df[(df['Brand_IV'] < 3) | (df['Brand_Cannibal_Ratio'] > 20)]
    
    with col1:
        st.metric("🚀 Launch Ready", len(launch_ready))
    with col2:
        st.metric("🧪 Pilot Ready", len(pilot_ready))
    with col3:
        st.metric("⏸️ Defer", len(defer))
    with col4:
        total_investment = len(launch_ready) + len(pilot_ready)
        st.metric("Total Investment", total_investment)
    
    st.markdown("---")
    
    # Top 5 launch ready SKUs
    st.markdown("### 🚀 TOP 5 LAUNCH READY SKUs")
    
    top_5_launch = launch_ready.head(5) if len(launch_ready) >= 5 else launch_ready
    
    for idx, (_, row) in enumerate(top_5_launch.iterrows(), 1):
        st.markdown(f"#### #{idx} {row['SKU']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Key Metrics**")
            st.caption(f"💰 IV: {row['Brand_IV']:.1f}%")
            st.caption(f"🎯 Capture: {row['Brand_Capture_Eff']:.1f}%")
            st.caption(f"♻️ Cannibal: {row['Brand_Cannibal_Ratio']:.1f}%")
            st.caption(f"🛡️ VaR: {row['Brand_VaR10']:.1f}%")
        
        with col2:
            st.markdown("**🎯 Launch Strategy**")
            st.caption("• **Timing**: Immediate launch")
            st.caption("• **Investment**: High priority")
            st.caption("• **Risk**: Low to moderate")
            st.caption("• **Expected ROI**: High")
        
        with col3:
            st.markdown("**📈 Success Factors**")
            st.caption(f"• Strong IV ({row['Brand_IV']:.1f}%)")
            st.caption(f"• High capture efficiency ({row['Brand_Capture_Eff']:.1f}%)")
            st.caption(f"• Low cannibalization ({row['Brand_Cannibal_Ratio']:.1f}%)")
            st.caption(f"• Manageable risk (VaR: {row['Brand_VaR10']:.1f}%)")
        
        st.markdown("---")
    
    # Investment recommendations
    st.markdown("### 💰 INVESTMENT RECOMMENDATIONS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🚀 Immediate Launch (High Priority)**")
        if len(launch_ready) > 0:
            st.caption(f"• {len(launch_ready)} SKUs ready for immediate launch")
            st.caption("• Expected strong performance")
            st.caption("• Low risk profile")
            st.caption("• High confidence in success")
        else:
            st.caption("• No SKUs meet immediate launch criteria")
    
    with col2:
        st.markdown("**🧪 Pilot Testing (Medium Priority)**")
        if len(pilot_ready) > 0:
            st.caption(f"• {len(pilot_ready)} SKUs recommended for pilot testing")
            st.caption("• Moderate performance potential")
            st.caption("• Requires validation")
            st.caption("• Lower risk approach")
        else:
            st.caption("• No SKUs recommended for pilot testing")
    
    # Next steps
    st.markdown("### 📋 NEXT STEPS")
    st.markdown("**For C-Level Decision Making:**")
    st.caption("1. **Approve immediate launch** of top 5 SKUs")
    st.caption("2. **Allocate resources** for pilot testing of medium-priority SKUs")
    st.caption("3. **Review detailed analysis** using full dashboard")
    st.caption("4. **Monitor performance** post-launch")
    st.caption("5. **Adjust strategy** based on market response")
    
    # Navigation
    st.markdown("---")



def create_closest_rivals_widget(df):
    """
    Create a closest rivals widget with diversion percentages for competitive messaging planning
    """
    # Check if dataframe is empty
    if df.empty:
        st.warning("⚠️ No SKUs available for closest rivals analysis")
        return
    
    st.markdown("### 🎯 Closest Rivals Analysis")
    st.markdown("*Top competitors by diversion percentage for competitive messaging planning*")
    
    # Aggregate competitor data across all SKUs
    competitor_data = {}
    
    for _, row in df.iterrows():
        # Top 3 competitors for each SKU
        competitors = [
            {'name': row['Top_Competitor'], 'ratio': row['Top_Competitor_Ratio']},
            {'name': row['Second_Competitor'], 'ratio': row['Second_Competitor_Ratio']},
            {'name': row['Third_Competitor'], 'ratio': row['Third_Competitor_Ratio']}
        ]
        
        for comp in competitors:
            if comp['name'] and comp['ratio'] > 0:
                if comp['name'] not in competitor_data:
                    competitor_data[comp['name']] = {
                        'total_diversion': 0,
                        'sku_count': 0,
                        'max_diversion': 0,
                        'avg_diversion': 0
                    }
                
                competitor_data[comp['name']]['total_diversion'] += comp['ratio']
                competitor_data[comp['name']]['sku_count'] += 1
                competitor_data[comp['name']]['max_diversion'] = max(
                    competitor_data[comp['name']]['max_diversion'], 
                    comp['ratio']
                )
    
    # Calculate average diversion for each competitor
    for comp_name in competitor_data:
        competitor_data[comp_name]['avg_diversion'] = (
            competitor_data[comp_name]['total_diversion'] / 
            competitor_data[comp_name]['sku_count']
        )
    
    # Sort by total diversion (most threatening competitors first)
    sorted_competitors = sorted(
        competitor_data.items(), 
        key=lambda x: x[1]['total_diversion'], 
        reverse=True
    )
    
    # Display top 10 competitors
    top_competitors = sorted_competitors[:10]
    
    if top_competitors:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🏆 Top Competitors by Total Diversion")
            
            # Create competitor list with metrics
            for i, (comp_name, data) in enumerate(top_competitors, 1):
                col_rank, col_name, col_metrics = st.columns([1, 3, 2])
                
                with col_rank:
                    st.markdown(f"**#{i}**")
                
                with col_name:
                    st.markdown(f"**{comp_name}**")
                    st.caption(f"Affects {data['sku_count']} SKU{'s' if data['sku_count'] > 1 else ''}")
                
                with col_metrics:
                    st.metric("Total Diversion", f"{data['total_diversion']:.1f}%")
                    st.caption(f"Avg: {data['avg_diversion']:.1f}% | Max: {data['max_diversion']:.1f}%")
                
                st.markdown("---")
        
        with col2:
            st.markdown("#### 📊 Diversion Distribution")
            
            # Create donut chart for top 5 competitors
            if len(top_competitors) >= 5:
                top_5 = top_competitors[:5]
                others_total = sum(data['total_diversion'] for _, data in top_competitors[5:])
                
                labels = [comp[0] for comp in top_5]
                values = [comp[1]['total_diversion'] for comp in top_5]
                
                if others_total > 0:
                    labels.append("Others")
                    values.append(others_total)
                
                # Create donut chart
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    textinfo='label+percent',
                    textfont=dict(size=10),
                    marker=dict(
                        colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
                        line=dict(color='#000000', width=2)
                    )
                )])
                
                fig.update_layout(
                    title="Top Competitors by Diversion",
                    showlegend=True,
                    height=400,
                    margin=dict(t=50, b=50, l=50, r=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Competitive messaging insights
            st.markdown("#### 💬 Messaging Insights")
            
            if top_competitors:
                top_competitor = top_competitors[0]
                st.info(f"**Primary Target**: {top_competitor[0]} ({top_competitor[1]['total_diversion']:.1f}% total diversion)")
                
                if len(top_competitors) > 1:
                    second_competitor = top_competitors[1]
                    st.warning(f"**Secondary Target**: {second_competitor[0]} ({second_competitor[1]['total_diversion']:.1f}% total diversion)")
                
                # Calculate competitive intensity
                total_diversion = sum(data['total_diversion'] for _, data in top_competitors)
                avg_diversion = total_diversion / len(top_competitors)
                
                if avg_diversion > 15:
                    st.error("🔥 **High Competitive Intensity** - Strong messaging needed")
                elif avg_diversion > 10:
                    st.warning("⚡ **Moderate Competitive Intensity** - Targeted messaging recommended")
                else:
                    st.success("✅ **Low Competitive Intensity** - Standard messaging sufficient")
        
        # Competitive messaging recommendations
        st.markdown("#### 🎯 Competitive Messaging Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎯 Primary Messaging**")
            if top_competitors:
                primary = top_competitors[0]
                st.caption(f"Focus on **{primary[0]}**")
                st.caption(f"• Highlight superior value proposition")
                st.caption(f"• Address {primary[0]}'s weaknesses")
                st.caption(f"• Target {primary[1]['sku_count']} affected SKUs")
        
        with col2:
            st.markdown("**📢 Secondary Messaging**")
            if len(top_competitors) > 1:
                secondary = top_competitors[1]
                st.caption(f"Address **{secondary[0]}**")
                st.caption(f"• Differentiate on key attributes")
                st.caption(f"• Emphasize unique benefits")
                st.caption(f"• Target {secondary[1]['sku_count']} affected SKUs")
            else:
                st.caption("No secondary competitor identified")
        
        with col3:
            st.markdown("**📊 Portfolio Strategy**")
            total_affected_skus = sum(data['sku_count'] for _, data in top_competitors)
            st.caption(f"**{total_affected_skus}** SKUs face competition")
            st.caption("• Coordinate messaging across portfolio")
            st.caption("• Avoid cannibalization between SKUs")
            st.caption("• Leverage brand owner strength")
        
        # Assumption note
        st.caption("📝 **Assumption**: Competitor analysis based on diversion ratios from market research. Messaging recommendations assume competitive positioning opportunities exist.")
    
    else:
        st.warning("⚠️ No competitor data available for analysis")


def create_definitions_thresholds_sheet():
    """
    Create a single-page definitions and thresholds sheet for quick reference
    """
    st.markdown("---")
    st.markdown("## 📋 Quick Reference: Definitions & Thresholds")
    st.markdown("*Single-page guide to all metrics, definitions, and decision thresholds*")
    
    # Create a compact layout with multiple columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Core Metrics")
        
        st.markdown("**💰 Incremental Value (IV)**")
        st.caption("**Definition**: Expected incremental sales volume from new SKU")
        st.caption("**🟢 GO**: ≥6.0% | **🟡 WATCH**: 4.0-5.9% | **🔴 HOLD**: <4.0%")
        
        st.markdown("**🎯 Capture Efficiency**")
        st.caption("**Definition**: % of volume captured from competitors vs own portfolio")
        st.caption("**🟢 GO**: ≥80.0% | **🟡 WATCH**: 70.0-79.9% | **🔴 HOLD**: <70.0%")
        
        st.markdown("**♻️ Cannibalization Ratio**")
        st.caption("**Definition**: % of volume taken from existing brand portfolio")
        st.caption("**🟢 GO**: ≤10.0% | **🟡 WATCH**: 10.0-20.0% | **🔴 HOLD**: >20.0%")
        
        st.markdown("**🛡️ VaR10 (Value at Risk)**")
        st.caption("**Definition**: 10th percentile of potential volume loss")
        st.caption("**🟢 GO**: ≤4.0% | **🟡 WATCH**: 4.0-6.0% | **🔴 HOLD**: >6.0%")
        
        st.markdown("**🌊 Category Lift**")
        st.caption("**Definition**: Additional volume from category expansion")
        st.caption("**🚀 High**: ≥6.0% | **📈 Moderate**: 4.0-5.9% | **📊 Low**: 2.0-3.9% | **⚠️ Minimal**: <2.0%")
    
    with col2:
        st.markdown("### 🎯 Decision Thresholds")
        
        st.markdown("**Launch Readiness**")
        st.caption("**Launch with Confidence**: IV≥5.0%, Capture≥70.0%, Cannibal≤15.0%, VaR≤5.0%")
        st.caption("**Pilot Recommended**: IV≥4.0%, Capture≥65.0%, Cannibal≤20.0%, VaR≤6.0%")
        st.caption("**Defer Launch**: Any metric below pilot thresholds")
        
        st.markdown("**Risk Levels**")
        st.caption("**Low Risk**: VaR≤4.0% + IV≥5.0%")
        st.caption("**Medium Risk**: VaR 4.0-6.0% OR IV 4.0-5.0%")
        st.caption("**High Risk**: VaR>6.0% OR IV<4.0%")
        
        st.markdown("**Performance Tiers**")
        st.caption("**High Performers**: Composite Score ≥75th percentile")
        st.caption("**Medium Performers**: Composite Score 25th-75th percentile")
        st.caption("**Low Performers**: Composite Score <25th percentile")
        
        st.markdown("**Sensitivity Levels**")
        st.caption("**High Sensitivity**: 3+ high-sensitivity factors")
        st.caption("**Moderate Sensitivity**: 2 high-sensitivity factors")
        st.caption("**Low Sensitivity**: 0-1 high-sensitivity factors")
    
    with col3:
        st.markdown("### 📈 Portfolio Benchmarks")
        
        st.markdown("**Market Concentration**")
        st.caption("**Low**: HHI <0.15 (competitive market)")
        st.caption("**Moderate**: HHI 0.15-0.25 (moderate concentration)")
        st.caption("**High**: HHI >0.25 (concentrated market)")
        
        st.markdown("**Competitive Position**")
        st.caption("**Strong**: Top competitor <8.0% diversion")
        st.caption("**Moderate**: Top competitor 8.0-12.0% diversion")
        st.caption("**Weak**: Top competitor >12.0% diversion")
        
        st.markdown("**Volume Sources**")
        st.caption("**Captured**: Volume from competitors (desirable)")
        st.caption("**Cannibalized**: Volume from own portfolio (monitor)")
        st.caption("**Category Lift**: Volume from market growth (bonus)")
    
    # Add formula section
    st.markdown("---")
    st.markdown("### 🧮 Key Formulas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Composite Score**")
        st.code("Score = (IV × 0.4) + (Capture × 0.4) + ((100-Cannibal) × 0.2)")
        
        st.markdown("**Net Effect (Brand Owner)**")
        st.code("Net = Owner_IV - (Owner_Cannibal × 0.2)")
        
        st.markdown("**True Expansion**")
        st.code("True_Expansion = Category_Lift - (Cannibal × 0.3)")
    
    with col2:
        st.markdown("**Risk Score**")
        st.code("Risk = (VaR10 × 0.6) + ((100-Prob_IV) × 0.4)")
        
        st.markdown("**Sensitivity Count**")
        st.code("High_Sensitivity = (IV≥6) + (Capture≥80) + (Cannibal≥15) + (VaR≥5)")
        
        st.markdown("**Confidence Score**")
        st.code("Confidence = min(100, (IV×10) + (Capture×0.5) + ((100-Cannibal)×0.5))")
    
    # Add quick decision matrix
    st.markdown("---")
    st.markdown("### ⚡ Quick Decision Matrix")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🟢 GO ZONE**")
        st.caption("• 💰 IV ≥ 6%")
        st.caption("• 🎯 Capture ≥ 80%")
        st.caption("• ♻️ Cannibal ≤ 10%")
        st.caption("• 🛡️ VaR ≤ 4%")
        st.caption("**→ Launch with confidence**")
    
    with col2:
        st.markdown("**🟡 CAUTION ZONE**")
        st.caption("• 💰 IV 4-6%")
        st.caption("• 🎯 Capture 70-80%")
        st.caption("• ♻️ Cannibal 10-20%")
        st.caption("• 🛡️ VaR 4-6%")
        st.caption("**→ Pilot test first**")
    
    with col3:
        st.markdown("**🔴 STOP ZONE**")
        st.caption("• 💰 IV < 4%")
        st.caption("• 🎯 Capture < 70%")
        st.caption("• ♻️ Cannibal > 20%")
        st.caption("• 🛡️ VaR > 6%")
        st.caption("**→ Defer or redesign**")
    
    # Add print note
    st.markdown("---")
    st.info("📄 **Print Note**: This page is optimized for single-page printing. Use 'Print to PDF' for best results.")

def create_cannibalization_watchlist(df):
    """
    Create cannibalization watchlist table showing internal impact at brand owner level
    """
    # Check if dataframe is empty
    if df.empty:
        st.warning("⚠️ No SKUs available for cannibalization analysis")
        return
    
    # Sort by cannibalization ratio (highest first)
    watchlist_df = df.sort_values('Owner_Cannibal_Ratio', ascending=False).copy()
    
    # Add risk level based on cannibalization
    def get_cannibal_risk_level(ratio):
        if ratio >= 20:
            return "🔴 High Risk"
        elif ratio >= 15:
            return "🟡 Medium Risk"
        else:
            return "🟢 Low Risk"
    
    watchlist_df['Risk_Level'] = watchlist_df['Owner_Cannibal_Ratio'].apply(get_cannibal_risk_level)
    
    # Create the watchlist table
    st.subheader("⚠️ Cannibalization Watchlist")
    st.markdown("*SKUs ranked by internal impact on brand owner portfolio*")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if len(df) > 0:
            st.metric("Avg Cannibalization", f"{df['Owner_Cannibal_Ratio'].mean():.1f}%")
        else:
            st.metric("Avg Cannibalization", "N/A")
    with col2:
        high_cannibal = len(df[df['Owner_Cannibal_Ratio'] >= 20])
        if len(df) > 0:
            st.metric("High Risk SKUs", high_cannibal, delta=f"{high_cannibal/len(df)*100:.0f}%")
        else:
            st.metric("High Risk SKUs", 0)
    with col3:
        if len(df) > 0:
            median_cannibal = df['Owner_Cannibal_Ratio'].median()
            st.metric("Median Impact", f"{median_cannibal:.1f}%")
        else:
            st.metric("Median Impact", "N/A")
    with col4:
        if len(df) > 0:
            max_cannibal = df['Owner_Cannibal_Ratio'].max()
            st.metric("Highest Impact", f"{max_cannibal:.1f}%")
        else:
            st.metric("Highest Impact", "N/A")
    
    # Create detailed table
    st.markdown("### 📊 Detailed Cannibalization Analysis")
    
    # Create table with key metrics
    for idx, (_, row) in enumerate(watchlist_df.iterrows()):
        with st.container():
            st.markdown("---")
            
            # First row: Basic info and metrics
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                st.subheader(f"{row['SKU']}")
                st.markdown(f"Rank: #{idx + 1}")
                st.markdown(f"**{row['Risk_Level']}**")
            
            with col2:
                st.subheader("Cannibalization Impact:")
                st.metric("Owner Cannibal Ratio", f"{row['Owner_Cannibal_Ratio']:.1f}%")
                st.metric("Brand Cannibal Ratio", f"{row['Brand_Cannibal_Ratio']:.1f}%")
                
                # Visual indicator
                cannibal_pct = row['Owner_Cannibal_Ratio']
                if cannibal_pct >= 20:
                    st.error(f"⚠️ High internal impact: {cannibal_pct:.1f}%")
                elif cannibal_pct >= 15:
                    st.warning(f"⚠️ Moderate internal impact: {cannibal_pct:.1f}%")
                else:
                    st.success(f"✅ Low internal impact: {cannibal_pct:.1f}%")
            
            with col3:
                st.subheader("Owner Portfolio Impact:")
                st.metric("Owner IV", f"{row['Owner_IV']:.1f}%")
                st.metric("Owner Capture Eff", f"{row['Owner_Capture_Eff']:.1f}%")
                
                # Net effect calculation
                net_effect = row['Owner_IV'] - (row['Owner_Cannibal_Ratio'] * 0.2)
                if net_effect >= 2.0:
                    st.success(f"Net Positive: +{net_effect:.1f}%")
                elif net_effect >= 0.5:
                    st.warning(f"Net Neutral: +{net_effect:.1f}%")
                else:
                    st.error(f"Net Negative: {net_effect:.1f}%")
            
            # Second row: Recommendation in 3 columns
            st.subheader("Recommendation:")
            rec_col1, rec_col2, rec_col3 = st.columns([1, 2, 2])
            
            with rec_col1:
                recommendation, confidence, confidence_score, rec_color = get_action_recommendation(row)
                
                if rec_color == "success":
                    st.success(recommendation)
                elif rec_color == "warning":
                    st.warning(recommendation)
                else:
                    st.error(recommendation)
                
                st.caption(f"Confidence: {confidence}")
                
                # Specific cannibalization advice
                if row['Owner_Cannibal_Ratio'] >= 20:
                    st.error("🚨 **Monitor closely** - High cannibalization risk")
                elif row['Owner_Cannibal_Ratio'] >= 15:
                    st.warning("⚠️ **Watch impact** - Moderate cannibalization")
                else:
                    st.success("✅ **Safe to proceed** - Low cannibalization")
            
            with rec_col2:
                # Add What Could Change box for each SKU
                create_what_could_change_box(row)
            
            with rec_col3:
                st.markdown("**💡 Recommendations:**")
                # Calculate sensitivity level for recommendations
                high_sensitivity_count = sum([
                    row['Brand_IV'] >= 6.0,
                    row['Brand_Capture_Eff'] >= 80,
                    row['Brand_Cannibal_Ratio'] >= 15,
                    row['Brand_VaR10'] >= 5
                ])
                
                # Add specific recommendations based on sensitivity
                if high_sensitivity_count >= 3:
                    st.caption("• Monitor price elasticity closely during launch")
                    st.caption("• Test pack formats in pilot markets")
                    st.caption("• Develop contingency plans for market changes")
                    st.caption("• Consider phased rollout to manage risk")
                elif high_sensitivity_count >= 2:
                    st.caption("• Monitor key sensitivity factors")
                    st.caption("• Prepare for potential market adjustments")
                    st.caption("• Consider A/B testing for optimization")
                else:
                    st.caption("• Relatively stable launch profile")
                    st.caption("• Focus on execution excellence")
                    st.caption("• Monitor for unexpected market shifts")
        
        # Add assumption note
        st.caption(get_assumption_note('cannibalization'))

def create_category_lift_table(df):
    """
    Create category lift table that ranks SKUs by lift to temper expectations on true expansion
    """
    # Check if dataframe is empty
    if df.empty:
        st.warning("⚠️ No SKUs available for category lift analysis")
        return
    
    # Sort by category lift ratio (highest first)
    lift_df = df.sort_values('Brand_Category_Lift', ascending=False).copy()
    
    # Add expansion potential level
    def get_expansion_potential(lift):
        if lift >= 6:
            return "🚀 High Expansion"
        elif lift >= 4:
            return "📈 Moderate Expansion"
        elif lift >= 2:
            return "📊 Low Expansion"
        else:
            return "⚠️ Minimal Expansion"
    
    lift_df['Expansion_Potential'] = lift_df['Brand_Category_Lift'].apply(get_expansion_potential)
    
    # Create the category lift table
    st.subheader("📈 Category Lift Analysis")
    st.markdown("*SKUs ranked by category expansion potential - temper expectations on true market growth*")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if len(df) > 0:
            st.metric("Avg Category Lift", f"{df['Brand_Category_Lift'].mean():.1f}%")
        else:
            st.metric("Avg Category Lift", "N/A")
    with col2:
        high_lift = len(df[df['Brand_Category_Lift'] >= 6])
        if len(df) > 0:
            st.metric("High Expansion SKUs", high_lift, delta=f"{high_lift/len(df)*100:.0f}%")
        else:
            st.metric("High Expansion SKUs", 0)
    with col3:
        if len(df) > 0:
            median_lift = df['Brand_Category_Lift'].median()
            st.metric("Median Lift", f"{median_lift:.1f}%")
        else:
            st.metric("Median Lift", "N/A")
    with col4:
        if len(df) > 0:
            max_lift = df['Brand_Category_Lift'].max()
            st.metric("Highest Lift", f"{max_lift:.1f}%")
        else:
            st.metric("Highest Lift", "N/A")
    
    # Reality check warning
    st.warning("⚠️ **Reality Check**: Category lift measures potential market expansion, but actual growth may be lower due to competitive response and market saturation.")
    
    # Create detailed table
    st.markdown("### 📊 Detailed Category Expansion Analysis")
    
    # Create table with key metrics
    for idx, (_, row) in enumerate(lift_df.iterrows()):
        with st.container():
            st.markdown("---")
            
            # First row: Basic info and metrics
            col1, col2, col3 = st.columns([2, 3, 3])
            
            with col1:
                st.subheader(f"{row['SKU']}")
                st.markdown(f"Rank: #{idx + 1}")
                st.markdown(f"**{row['Expansion_Potential']}**")
            
            with col2:
                st.subheader("Category Expansion Metrics:")
                st.metric("Brand Category Lift", f"{row['Brand_Category_Lift']:.1f}%")
                st.metric("Owner Category Lift", f"{row['Owner_Category_Lift']:.1f}%")
                
                # Visual indicator
                lift_pct = row['Brand_Category_Lift']
                if lift_pct >= 6:
                    st.success(f"🚀 Strong expansion potential: {lift_pct:.1f}%")
                elif lift_pct >= 4:
                    st.info(f"📈 Moderate expansion potential: {lift_pct:.1f}%")
                elif lift_pct >= 2:
                    st.warning(f"📊 Limited expansion potential: {lift_pct:.1f}%")
                else:
                    st.error(f"⚠️ Minimal expansion potential: {lift_pct:.1f}%")
            
            with col3:
                st.subheader("Volume Source Breakdown:")
                st.metric("Captured Volume", f"{row['Brand_Abs_Captured']:.1f}%")
                st.metric("Cannibalized Volume", f"{row['Brand_Abs_Cannibalized']:.1f}%")
                st.metric("Category Lift Volume", f"{row['Brand_Abs_Category_Lift']:.1f}%")
                
                # True expansion calculation (category lift vs cannibalization)
                true_expansion = row['Brand_Category_Lift'] - (row['Brand_Cannibal_Ratio'] * 0.3)
                if true_expansion >= 5:
                    st.success(f"True Expansion: +{true_expansion:.1f}%")
                elif true_expansion >= 2:
                    st.warning(f"Mixed Impact: +{true_expansion:.1f}%")
                else:
                    st.error(f"Limited Expansion: {true_expansion:.1f}%")
            
            # Second row: Headers on same level
            exp_col1, exp_col2 = st.columns([1, 4])
            
            with exp_col1:
                st.subheader("Expectation Management:")
            
            with exp_col2:
                st.subheader("🔄 What Could Change?")
            
            # Third row: Content below headers
            exp_col1, exp_col2 = st.columns([1, 4])
            
            with exp_col1:
                # Tempered expectations based on lift level
                if row['Brand_Category_Lift'] >= 6:
                    st.success("🎯 **High Expectations** - Strong category growth potential")
                    st.caption("Monitor competitive response")
                elif row['Brand_Category_Lift'] >= 4:
                    st.info("📊 **Moderate Expectations** - Some category expansion")
                    st.caption("Expect gradual growth")
                elif row['Brand_Category_Lift'] >= 2:
                    st.warning("⚠️ **Low Expectations** - Limited true expansion")
                    st.caption("Focus on market share")
                else:
                    st.error("🚫 **Minimal Expectations** - Little category growth")
                    st.caption("Volume from share shift")
                
                # Additional context
                if row['Brand_Cannibal_Ratio'] > row['Brand_Category_Lift']:
                    st.error("⚠️ More cannibalization than expansion")
                elif row['Brand_Category_Lift'] > row['Brand_Cannibal_Ratio'] * 2:
                    st.success("✅ True expansion potential")
                else:
                    st.warning("⚖️ Balanced impact")
            
            with exp_col2:
                # Add What Could Change box for each SKU (without header since it's already above)
                # We need to modify create_what_could_change_box to not show the header
                # For now, let's create a custom version
                high_sensitivity_count = sum([
                    row['Brand_IV'] >= 6.0,
                    row['Brand_Capture_Eff'] >= 80,
                    row['Brand_Cannibal_Ratio'] >= 15,
                    row['Brand_VaR10'] >= 5
                ])
                
                if high_sensitivity_count >= 3:
                    sensitivity_level = "🔴 High Sensitivity"
                    sensitivity_color = "error"
                elif high_sensitivity_count >= 2:
                    sensitivity_level = "🟡 Moderate Sensitivity"
                    sensitivity_color = "warning"
                else:
                    sensitivity_level = "🟢 Low Sensitivity"
                    sensitivity_color = "success"
                
                # Display sensitivity level
                if sensitivity_color == "success":
                    st.success(f"**{sensitivity_level}**")
                elif sensitivity_color == "warning":
                    st.warning(f"**{sensitivity_level}**")
                else:
                    st.error(f"**{sensitivity_level}**")
                
                # Display sensitivity factors
                sensitivity_factors = []
                
                # Price sensitivity
                if row['Brand_IV'] >= 6.0:
                    sensitivity_factors.append("💰 **Price Sensitive**: High IV suggests strong price elasticity - 10% price increase could reduce demand by 15-20%")
                elif row['Brand_IV'] >= 4.0:
                    sensitivity_factors.append("💰 **Moderate Price Sensitivity**: Some IV suggests moderate price elasticity - price changes could impact 5-10% of volume")
                else:
                    sensitivity_factors.append("💰 **Low Price Sensitivity**: Lower IV suggests limited price elasticity - price changes have minimal impact")
                
                # Pack sensitivity
                if row['Brand_Capture_Eff'] >= 80:
                    sensitivity_factors.append("📦 **Moderate Pack Sensitivity**: Good capture efficiency suggests some pack-size awareness - format changes could impact 10-20% of volume")
                elif row['Brand_Capture_Eff'] >= 60:
                    sensitivity_factors.append("📦 **Low Pack Sensitivity**: Moderate capture efficiency suggests limited pack-size awareness - format changes have minimal impact")
                else:
                    sensitivity_factors.append("📦 **Very Low Pack Sensitivity**: Lower capture efficiency suggests minimal pack-size awareness - format changes unlikely to impact volume")
                
                # Cannibalization sensitivity
                if row['Brand_Cannibal_Ratio'] >= 15:
                    sensitivity_factors.append("⚠️ **High Cannibalization Risk**: Changes could significantly impact internal portfolio - monitor brand owner portfolio closely")
                elif row['Brand_Cannibal_Ratio'] >= 8:
                    sensitivity_factors.append("⚠️ **Moderate Cannibalization Risk**: Changes could affect some internal products - watch for portfolio conflicts")
                else:
                    sensitivity_factors.append("✅ **Low Cannibalization Risk**: Changes unlikely to significantly impact internal portfolio")
                
                # Category lift sensitivity
                if row['Brand_Category_Lift'] >= 6:
                    sensitivity_factors.append("📈 **Category Growth Dependent**: High lift suggests reliance on category expansion - market conditions could significantly impact performance")
                elif row['Brand_Category_Lift'] >= 4:
                    sensitivity_factors.append("📈 **Moderate Category Dependency**: Some lift suggests partial reliance on category growth - monitor market trends")
                else:
                    sensitivity_factors.append("📈 **Low Category Dependency**: Lower lift suggests more share-based growth - less sensitive to market conditions")
                
                # Risk sensitivity
                if row['Brand_VaR10'] >= 5:
                    sensitivity_factors.append("🎯 **High Risk Sensitivity**: Elevated VaR suggests high volatility - external factors could cause significant swings")
                elif row['Brand_VaR10'] >= 4:
                    sensitivity_factors.append("🎯 **Moderate Risk Sensitivity**: Some VaR suggests moderate volatility - monitor external factors")
                else:
                    sensitivity_factors.append("🎯 **Low Risk Sensitivity**: Lower VaR suggests stability - less sensitive to external shocks")
                
                # Display sensitivity factors
                for factor in sensitivity_factors:
                    st.caption(factor)
    
    # Portfolio insights
    st.markdown("---")
    st.subheader("📋 Portfolio Category Expansion Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Expansion Distribution:**")
        high_expansion = len(df[df['Brand_Category_Lift'] >= 6])
        moderate_expansion = len(df[(df['Brand_Category_Lift'] >= 4) & (df['Brand_Category_Lift'] < 6)])
        low_expansion = len(df[(df['Brand_Category_Lift'] >= 2) & (df['Brand_Category_Lift'] < 4)])
        minimal_expansion = len(df[df['Brand_Category_Lift'] < 2])
        
        st.write(f"🚀 High Expansion (≥6%): {high_expansion} SKUs")
        st.write(f"📈 Moderate Expansion (4-6%): {moderate_expansion} SKUs")
        st.write(f"📊 Low Expansion (2-4%): {low_expansion} SKUs")
        st.write(f"⚠️ Minimal Expansion (<2%): {minimal_expansion} SKUs")
    
    with col2:
        st.markdown("**Key Insights:**")
        avg_lift = df['Brand_Category_Lift'].mean()
        avg_cannibal = df['Brand_Cannibal_Ratio'].mean()
        
        if avg_lift > avg_cannibal:
            st.success(f"✅ Portfolio shows net expansion potential (+{avg_lift-avg_cannibal:.1f}%)")
        else:
            st.warning(f"⚠️ Portfolio shows net cannibalization risk ({avg_cannibal-avg_lift:.1f}%)")
        
        st.write(f"📊 Average category lift: {avg_lift:.1f}%")
        st.write(f"📊 Average cannibalization: {avg_cannibal:.1f}%")
        
        # Top expansion opportunity
        top_lift_sku = df.loc[df['Brand_Category_Lift'].idxmax()]
        st.write(f"🎯 Best expansion opportunity: {top_lift_sku['SKU']} ({top_lift_sku['Brand_Category_Lift']:.1f}%)")
        
        # Add assumption note
        st.caption(get_assumption_note('category_lift'))

def create_kpi_strip(df):
    """
    Create a KPI strip at the bottom that repeats key numbers for scanning when printed
    """
    # Check if dataframe is empty
    if df.empty:
        st.warning("⚠️ No SKUs available for KPI summary")
        return
    
    st.markdown("---")
    st.markdown("### 📊 Portfolio KPI Summary - Quick Reference")
    st.markdown("*Key metrics for executive scanning and printing*")
    
    # Create multiple rows of KPIs for comprehensive coverage
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total SKUs", len(df))
        st.caption("Portfolio Size")
    
    with col2:
        avg_iv = df['Brand_IV'].mean()
        st.metric("Avg Brand IV", f"{avg_iv:.1f}%")
        st.caption("Incremental Value")
    
    with col3:
        avg_capture = df['Brand_Capture_Eff'].mean()
        st.metric("Avg Capture", f"{avg_capture:.1f}%")
        st.caption("Efficiency")
    
    with col4:
        avg_cannibal = df['Brand_Cannibal_Ratio'].mean()
        st.metric("Avg Cannibal", f"{avg_cannibal:.1f}%")
        st.caption("Internal Impact")
    
    with col5:
        avg_lift = df['Brand_Category_Lift'].mean()
        st.metric("Avg Category Lift", f"{avg_lift:.1f}%")
        st.caption("Expansion")
    
    with col6:
        avg_var = df['Brand_VaR10'].mean()
        st.metric("Avg VaR10", f"{avg_var:.1f}%")
        st.caption("Risk Level")
    
    # Second row - Portfolio distribution
    st.markdown("#### 📈 Portfolio Distribution")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        high_iv = len(df[df['Brand_IV'] >= 6])
        delta_pct = f"{high_iv/len(df)*100:.0f}%" if len(df) > 0 else "0%"
        st.metric("High IV SKUs", high_iv, delta=delta_pct)
        st.caption("≥6.0% Incremental Value")
    
    with col2:
        high_capture = len(df[df['Brand_Capture_Eff'] >= 80])
        delta_pct = f"{high_capture/len(df)*100:.0f}%" if len(df) > 0 else "0%"
        st.metric("High Capture", high_capture, delta=delta_pct)
        st.caption("≥80.0% Efficiency")
    
    with col3:
        low_cannibal = len(df[df['Brand_Cannibal_Ratio'] <= 10])
        delta_pct = f"{low_cannibal/len(df)*100:.0f}%" if len(df) > 0 else "0%"
        st.metric("Low Cannibal", low_cannibal, delta=delta_pct)
        st.caption("≤10.0% Impact")
    
    with col4:
        high_lift = len(df[df['Brand_Category_Lift'] >= 6])
        delta_pct = f"{high_lift/len(df)*100:.0f}%" if len(df) > 0 else "0%"
        st.metric("High Expansion", high_lift, delta=delta_pct)
        st.caption("≥6.0% Category Lift")
    
    with col5:
        low_risk = len(df[df['Brand_VaR10'] <= 4])
        delta_pct = f"{low_risk/len(df)*100:.0f}%" if len(df) > 0 else "0%"
        st.metric("Low Risk", low_risk, delta=delta_pct)
        st.caption("≤4.0% VaR10")
    
    with col6:
        top_skus = len(df[df['Composite_Score'] >= df['Composite_Score'].quantile(0.8)])
        delta_pct = f"{top_skus/len(df)*100:.0f}%" if len(df) > 0 else "0%"
        st.metric("Top Performers", top_skus, delta=delta_pct)
        st.caption("Top 20% Score")
    
    # Third row - Key insights and recommendations
    st.markdown("#### 🎯 Key Portfolio Insights")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_sku = df.loc[df['Composite_Score'].idxmax()]
        st.metric("Best Overall", best_sku['SKU'])
        st.caption(f"Score: {best_sku['Composite_Score']:.1f}")
    
    with col2:
        highest_iv = df.loc[df['Brand_IV'].idxmax()]
        st.metric("Highest IV", highest_iv['SKU'])
        st.caption(f"IV: {highest_iv['Brand_IV']:.1f}%")
    
    with col3:
        lowest_cannibal = df.loc[df['Brand_Cannibal_Ratio'].idxmin()]
        st.metric("Lowest Cannibal", lowest_cannibal['SKU'])
        st.caption(f"Cannibal: {lowest_cannibal['Brand_Cannibal_Ratio']:.1f}%")
    
    with col4:
        highest_lift = df.loc[df['Brand_Category_Lift'].idxmax()]
        st.metric("Highest Lift", highest_lift['SKU'])
        st.caption(f"Lift: {highest_lift['Brand_Category_Lift']:.1f}%")
    
    # Fourth row - Risk and opportunity summary
    st.markdown("#### ⚠️ Risk & Opportunity Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk = len(df[df['Brand_VaR10'] > 6])
        st.metric("High Risk SKUs", high_risk)
        st.caption("VaR10 > 6.0%")
    
    with col2:
        high_cannibal = len(df[df['Brand_Cannibal_Ratio'] > 20])
        st.metric("High Cannibal", high_cannibal)
        st.caption("Cannibal > 20.0%")
    
    with col3:
        launch_ready = len(df[(df['Brand_IV'] >= 5) & (df['Brand_Capture_Eff'] >= 70) & (df['Brand_Cannibal_Ratio'] <= 15)])
        st.metric("Launch Ready", launch_ready)
        st.caption("IV≥5.0%, Cap≥70.0%, Can≤15.0%")
    
    with col4:
        watch_list = len(df[(df['Brand_VaR10'] > 5) | (df['Brand_Cannibal_Ratio'] > 15)])
        st.metric("Watch List", watch_list)
        st.caption("High Risk or Cannibal")
    
    # Footer with timestamp and version
    st.markdown("---")
    st.markdown("**Notes**")
    st.caption("💡 Use this KPI strip for quick portfolio assessment and executive briefings")
    st.caption("📊 **Standardized Scales**: All charts use consistent scales for reliable visual comparisons")
    st.caption("♿ **Accessibility**: Design uses shapes, text, and position instead of color for print and accessibility")
    st.caption("📋 **Data Notes**: All analyses based on 6 SKU portfolio with standardized modeling assumptions and confidence intervals")

def create_small_multiples(df):
    """
    Create small multiples: identical micro-charts per metric across all SKUs on one spread
    """
    # Check if dataframe is empty
    if df.empty:
        st.warning("⚠️ No SKUs available for small multiples analysis")
        return
    
    st.subheader("📊 Small Multiples - Portfolio Comparison")
    st.markdown("*Identical micro-charts for each metric across all SKUs - perfect for side-by-side comparison*")
    
    # Define key metrics for small multiples
    metrics = [
        {'name': 'Brand_IV', 'title': 'Incremental Value', 'range': [0, 8], 'unit': '%', 'higher_better': True},
        {'name': 'Brand_Capture_Eff', 'title': 'Capture Efficiency', 'range': [0, 100], 'unit': '%', 'higher_better': True},
        {'name': 'Brand_Cannibal_Ratio', 'title': 'Cannibalization', 'range': [0, 25], 'unit': '%', 'higher_better': False},
        {'name': 'Brand_Category_Lift', 'title': 'Category Lift', 'range': [0, 8], 'unit': '%', 'higher_better': True},
        {'name': 'Brand_VaR10', 'title': 'VaR10 Risk', 'range': [0, 8], 'unit': '%', 'higher_better': False},
        {'name': 'Composite_Score', 'title': 'Composite Score', 'range': [0, 100], 'unit': '', 'higher_better': True}
    ]
    
    # Create small multiples for each metric
    for metric in metrics:
        st.markdown(f"### {metric['title']} - All SKUs")
        
        # Create columns for all SKUs
        cols = st.columns(len(df))
        
        for idx, (_, row) in enumerate(df.iterrows()):
            with cols[idx]:
                # Create mini bar chart for this SKU and metric
                value = row[metric['name']]
                
                # Calculate relative position within range
                min_val, max_val = metric['range']
                relative_height = (value - min_val) / (max_val - min_val) * 100
                relative_height = max(5, min(95, relative_height))  # Clamp between 5-95%
                
                # Calculate confidence interval for key metrics only
                confidence_visible = False
                confidence_height = 0
                
                if metric['name'] in ['Brand_IV', 'Brand_Capture_Eff', 'Brand_VaR10']:
                    confidence_visible = True
                    if metric['name'] == 'Brand_IV':
                        confidence_height = (row['Brand_IV_Prob'] / 100 * 0.3) / (max_val - min_val) * 100
                    elif metric['name'] == 'Brand_Capture_Eff':
                        confidence_height = (row['Brand_IV_Prob'] / 100 * 0.2) / (max_val - min_val) * 100
                    elif metric['name'] == 'Brand_VaR10':
                        confidence_height = ((100 - row['Brand_IV_Prob']) / 100 * 0.2) / (max_val - min_val) * 100
                    
                    confidence_height = max(2, min(10, confidence_height))  # Clamp between 2-10%
                
                # Use HTML visualization for ALL metrics (like VaR10 Risk section)
                # Determine color based on performance
                if (metric['higher_better'] and value >= df[metric['name']].quantile(0.75)) or \
                   (not metric['higher_better'] and value <= df[metric['name']].quantile(0.25)):
                    bar_color = '#2E8B57'  # Green
                elif (metric['higher_better'] and value >= df[metric['name']].median()) or \
                     (not metric['higher_better'] and value <= df[metric['name']].median()):
                    bar_color = '#FFA500'  # Orange
                else:
                    bar_color = '#DC143C'  # Red
                
                # Create HTML chart for all metrics
                chart_html = f"""
                <div style="text-align: center; margin: 5px;">
                    <div style="font-weight: bold; font-size: 12px; margin-bottom: 5px;">{row['SKU']}</div>
                    <div style="position: relative; width: 100%; height: 60px; border: 1px solid #ccc; background: #f9f9f9;">
                        <div style="position: absolute; bottom: 0; left: 0; width: 100%; height: {relative_height}%; 
                                    background: {bar_color}; border-top: 1px solid #333;"></div>
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                                    font-weight: bold; font-size: 11px; color: #333;">
                            {value:.1f}{metric['unit']}
                        </div>
                    </div>
                    <div style="font-size: 10px; color: #666; margin-top: 2px;">
                        {create_sparkbar(df, value, metric['name'], not metric['higher_better'])}
                    </div>
                </div>
                """
                
                st.markdown(chart_html, unsafe_allow_html=True)
        
        # Add summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average", f"{df[metric['name']].mean():.1f}{metric['unit']}")
        with col2:
            st.metric("Median", f"{df[metric['name']].median():.1f}{metric['unit']}")
        with col3:
            st.metric("Min", f"{df[metric['name']].min():.1f}{metric['unit']}")
        with col4:
            st.metric("Max", f"{df[metric['name']].max():.1f}{metric['unit']}")
        
        st.markdown("---")
        st.caption(get_assumption_note('small_multiples'))
    
    # Create a combined comparison matrix
    st.subheader("📋 Quick Comparison Matrix")
    st.markdown("*All metrics for all SKUs in one view*")
    
    # Create comparison table
    comparison_data = []
    for _, row in df.iterrows():
        comparison_data.append({
            'SKU': row['SKU'],
            'IV': f"{row['Brand_IV']:.1f}%",
            'Capture': f"{row['Brand_Capture_Eff']:.1f}%",
            'Cannibal': f"{row['Brand_Cannibal_Ratio']:.1f}%",
            'Lift': f"{row['Brand_Category_Lift']:.1f}%",
            'VaR10': f"{row['Brand_VaR10']:.1f}%",
            'Score': f"{row['Composite_Score']:.1f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Add interpretation guide
    st.markdown("---")
    st.subheader("📖 How to Read Small Multiples")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Color Coding:**")
        st.markdown("🟢 **Green**: Top 25% performance")
        st.markdown("🟠 **Orange**: Above median performance")
        st.markdown("🔴 **Red**: Below median performance")
    
    with col2:
        st.markdown("**Usage Tips:**")
        st.markdown("• Compare heights across SKUs for each metric")
        st.markdown("• Look for consistent patterns across metrics")
        st.markdown("• Identify outliers and underperformers")
        st.markdown("• Use for executive briefings and presentations")

def create_hierarchical_table(df):
    html = '<style>.hierarchical-table{border-collapse:collapse;width:100%;font-family:Arial;font-size:10px;margin:10px 0}.hierarchical-table th,.hierarchical-table td{border:1px solid #ddd;padding:4px;text-align:center;vertical-align:middle}.hierarchical-table th{background-color:#f2f2f2;font-weight:bold}.top-header{background-color:#e6e6e6}.sub-header{background-color:#f8f8f8}.sku-cell{background-color:#f9f9f9;font-weight:bold;text-align:left}</style><table class="hierarchical-table"><tr><th rowspan="2" class="top-header sku-cell">SKU</th><th colspan="3" class="top-header">Brand</th><th colspan="3" class="top-header">Brand Owner</th><th colspan="6" class="top-header">Brand Source of Volume</th><th colspan="6" class="top-header">Brand Owner Source of Volume</th><th colspan="5" class="top-header">Mean Diversion Ratio</th><th rowspan="2" class="top-header">Market Concentration (Diversion HHI)</th></tr><tr><th class="sub-header">Incremental Value</th><th class="sub-header">Probability of Incremental Value</th><th class="sub-header">VaR10</th><th class="sub-header">Incremental Value</th><th class="sub-header">Probability of Incremental Value</th><th class="sub-header">VaR10</th><th class="sub-header">Capture Efficiency</th><th class="sub-header">Cannibalization ratio</th><th class="sub-header">Category Lift Ratio</th><th class="sub-header">Average Absolute Captured</th><th class="sub-header">Average Absolute Cannibalized</th><th class="sub-header">Average Absolute Category Lift</th><th class="sub-header">Capture Efficiency</th><th class="sub-header">Cannibalization ratio</th><th class="sub-header">Category Lift Ratio</th><th class="sub-header">Average Absolute Captured</th><th class="sub-header">Average Absolute Cannibalized</th><th class="sub-header">Average Absolute Category Lift</th><th class="sub-header">Competition</th><th class="sub-header">%</th><th class="sub-header">Competition</th><th class="sub-header">%</th><th class="sub-header">Competition</th></tr>'
    
    for _, row in df.iterrows():
        html += f'<tr><td class="sku-cell">{row["SKU"]}</td><td>{row["Brand_IV"]:.1f}%</td><td>{row["Brand_IV_Prob"]:.1f}%</td><td>{row["Brand_VaR10"]:.1f}%</td><td>{row["Owner_IV"]:.1f}%</td><td>{row["Owner_IV_Prob"]:.1f}%</td><td>{row["Owner_VaR10"]:.1f}%</td><td>{row["Brand_Capture_Eff"]:.1f}%</td><td>{row["Brand_Cannibal_Ratio"]:.1f}%</td><td>{row["Brand_Category_Lift"]:.1f}%</td><td>{row["Brand_Abs_Captured"]:.1f}%</td><td>{row["Brand_Abs_Cannibalized"]:.1f}%</td><td>{row["Brand_Abs_Category_Lift"]:.1f}%</td><td>{row["Owner_Capture_Eff"]:.1f}%</td><td>{row["Owner_Cannibal_Ratio"]:.1f}%</td><td>{row["Owner_Category_Lift"]:.1f}%</td><td>{row["Owner_Abs_Captured"]:.1f}%</td><td>{row["Owner_Abs_Cannibalized"]:.1f}%</td><td>{row["Owner_Abs_Category_Lift"]:.1f}%</td><td>{row["Top_Competitor"]}</td><td>{row["Top_Competitor_Ratio"]:.1f}%</td><td>{row["Second_Competitor"]}</td><td>{row["Second_Competitor_Ratio"]:.1f}%</td><td>{row["Third_Competitor"]}</td><td>{row["Market_HHI"]:.2f}%</td></tr>'
    
    return html + '</table>'

def create_bookmark_button(sku_name, row):
    """Create a bookmark/star button for executives to mark SKUs for follow-up"""
    is_bookmarked = sku_name in st.session_state.bookmarked_skus
    
    # Use markdown to center the star with CSS
    st.markdown("""
        <style>
        div[data-testid*="stButton"] > button {
            width: 100%;
            text-align: center;
            padding: 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if is_bookmarked:
        if st.button("⭐", key=f"bookmark_{sku_name}", help="Remove from bookmarks"):
            st.session_state.bookmarked_skus.remove(sku_name)
            st.rerun()
    else:
        if st.button("☆", key=f"bookmark_{sku_name}", help="Add to bookmarks for follow-up"):
            st.session_state.bookmarked_skus.add(sku_name)
            st.rerun()
    
    return is_bookmarked

def create_bookmarked_skus_panel():
    """Create a panel showing all bookmarked SKUs"""
    if st.session_state.bookmarked_skus:
        st.sidebar.markdown("---")
        st.sidebar.subheader("⭐ Bookmarked SKUs")
        
        for sku in sorted(st.session_state.bookmarked_skus):
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.sidebar.write(f"• {sku}")
            with col2:
                if st.sidebar.button("🗑️", key=f"remove_{sku}", help="Remove bookmark"):
                    st.session_state.bookmarked_skus.remove(sku)
                    st.rerun()
        
        if st.sidebar.button("🗑️ Clear All Bookmarks"):
            st.session_state.bookmarked_skus.clear()
            st.rerun()
    else:
        st.sidebar.markdown("---")
        st.sidebar.caption("⭐ No bookmarked SKUs yet")

def create_simple_decision_page(df):
    """
    Create a simple decision page: "Launch now," "Pilot," or "Defer," listing the SKUs under each
    """
    # Check if DataFrame is empty
    if df.empty:
        st.warning("⚠️ No SKUs available for decision analysis. Please adjust your filters.")
        return
    
    st.markdown("---")
    st.markdown("## 🎯 Simple Decision Page")
    st.markdown("*Clear recommendations for each SKU based on performance analysis*")
    
    # Categorize SKUs based on action recommendations
    launch_now_skus = []
    pilot_skus = []
    defer_skus = []
    
    for idx, row in df.iterrows():
        recommendation, confidence, confidence_score, rec_color = get_action_recommendation(row)
        
        if rec_color == "success":
            launch_now_skus.append({
                'sku': row['SKU'],
                'score': row['Composite_Score'],
                'iv': row['Brand_IV'],
                'capture': row['Brand_Capture_Eff'],
                'cannibal': row['Brand_Cannibal_Ratio'],
                'var': row['Brand_VaR10'],
                'confidence': confidence,
                'confidence_score': confidence_score
            })
        elif rec_color == "warning":
            pilot_skus.append({
                'sku': row['SKU'],
                'score': row['Composite_Score'],
                'iv': row['Brand_IV'],
                'capture': row['Brand_Capture_Eff'],
                'cannibal': row['Brand_Cannibal_Ratio'],
                'var': row['Brand_VaR10'],
                'confidence': confidence,
                'confidence_score': confidence_score
            })
        else:
            defer_skus.append({
                'sku': row['SKU'],
                'score': row['Composite_Score'],
                'iv': row['Brand_IV'],
                'capture': row['Brand_Capture_Eff'],
                'cannibal': row['Brand_Cannibal_Ratio'],
                'var': row['Brand_VaR10'],
                'confidence': confidence,
                'confidence_score': confidence_score
            })
    
    # Sort each category by composite score (highest first)
    launch_now_skus.sort(key=lambda x: x['score'], reverse=True)
    pilot_skus.sort(key=lambda x: x['score'], reverse=True)
    defer_skus.sort(key=lambda x: x['score'], reverse=True)
    
    # First row: Launch Now (full width)
    st.markdown("### 🚀 Launch Now")
    st.markdown(f"**{len(launch_now_skus)} SKUs**")
    st.caption("High performance, low risk - ready for immediate launch")
    
    if launch_now_skus:
        # Create 5 columns for SKUs
        sku_cols = st.columns(5)
        
        for idx, sku_data in enumerate(launch_now_skus):
            col_idx = idx % 5  # Cycle through 5 columns
            
            with sku_cols[col_idx]:
                with st.container():
                    st.markdown(f"**{sku_data['sku']}**")
                    st.caption(f"Score: {sku_data['score']:.1f}")
                    st.caption(f"Confidence: {sku_data['confidence']}")
                    
                    # Key metrics in compact format
                    st.caption(f"💰 IV: {sku_data['iv']:.1f}%")
                    st.caption(f"🎯 Cap: {sku_data['capture']:.1f}%")
                    st.caption(f"♻️ Can: {sku_data['cannibal']:.1f}%")
                    st.caption(f"🛡️ VaR: {sku_data['var']:.1f}%")
                    
                    st.markdown("---")
    else:
        st.info("No SKUs ready for immediate launch")
    
    # Second row: Pilot and Defer in 2 columns
    col2, col3 = st.columns(2)
    
    with col2:
        st.markdown("### 🧪 Pilot")
        st.markdown(f"**{len(pilot_skus)} SKUs**")
        st.caption("Moderate performance - test in controlled environment first")
        
        if pilot_skus:
            for sku_data in pilot_skus:
                with st.container():
                    st.markdown(f"**{sku_data['sku']}**")
                    st.caption(f"Score: {sku_data['score']:.1f} | Confidence: {sku_data['confidence']}")
                    
                    # Key metrics in compact format
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        st.caption(f"💰 IV: {sku_data['iv']:.1f}%")
                        st.caption(f"🎯 Cap: {sku_data['capture']:.1f}%")
                    with col2_2:
                        st.caption(f"♻️ Can: {sku_data['cannibal']:.1f}%")
                        st.caption(f"🛡️ VaR: {sku_data['var']:.1f}%")
                    
                    st.markdown("---")
        else:
            st.info("No SKUs recommended for pilot testing")
    
    with col3:
        st.markdown("### ⏸️ Defer")
        st.markdown(f"**{len(defer_skus)} SKUs**")
        st.caption("Low performance or high risk - needs optimization before launch")
        
        if defer_skus:
            for sku_data in defer_skus:
                with st.container():
                    st.markdown(f"**{sku_data['sku']}**")
                    st.caption(f"Score: {sku_data['score']:.1f} | Confidence: {sku_data['confidence']}")
                    
                    # Key metrics in compact format
                    col3_1, col3_2 = st.columns(2)
                    with col3_1:
                        st.caption(f"💰 IV: {sku_data['iv']:.1f}%")
                        st.caption(f"🎯 Cap: {sku_data['capture']:.1f}%")
                    with col3_2:
                        st.caption(f"♻️ Can: {sku_data['cannibal']:.1f}%")
                        st.caption(f"🛡️ VaR: {sku_data['var']:.1f}%")
                    
                    st.markdown("---")
        else:
            st.info("No SKUs need to be deferred")
    
    # Summary statistics
    st.markdown("---")
    st.markdown("### 📊 Decision Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total SKUs", len(df))
    
    with col2:
        delta_pct = f"{len(launch_now_skus)/len(df)*100:.0f}%" if len(df) > 0 else "0%"
        st.metric("Launch Now", len(launch_now_skus), delta=delta_pct)
    
    with col3:
        delta_pct = f"{len(pilot_skus)/len(df)*100:.0f}%" if len(df) > 0 else "0%"
        st.metric("Pilot", len(pilot_skus), delta=delta_pct)
    
    with col4:
        delta_pct = f"{len(defer_skus)/len(df)*100:.0f}%" if len(df) > 0 else "0%"
        st.metric("Defer", len(defer_skus), delta=delta_pct)
    
    # Decision criteria explanation
    st.markdown("---")
    st.markdown("### 📋 Decision Criteria")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🚀 Launch Now**")
        st.caption("• High Incremental Value (≥5.0%)")
        st.caption("• High Capture Efficiency (≥70.0%)")
        st.caption("• Low Cannibalization (≤15.0%)")
        st.caption("• Low Risk (VaR ≤5.0%)")
        st.caption("• High Confidence (≥80/100)")
    
    with col2:
        st.markdown("**🧪 Pilot**")
        st.caption("• Moderate Incremental Value (≥4.0%)")
        st.caption("• Good Capture Efficiency (≥65.0%)")
        st.caption("• Moderate Cannibalization (≤20.0%)")
        st.caption("• Moderate Risk (VaR ≤6.0%)")
        st.caption("• Medium Confidence (60-79/100)")
    
    with col3:
        st.markdown("**⏸️ Defer**")
        st.caption("• Low Incremental Value (<4.0%)")
        st.caption("• Low Capture Efficiency (<65.0%)")
        st.caption("• High Cannibalization (>20.0%)")
        st.caption("• High Risk (VaR >6.0%)")
        st.caption("• Low Confidence (<60/100)")
    
    # Next steps
    st.markdown("---")
    st.markdown("### 🎯 Next Steps")
    
    if len(launch_now_skus) > 0:
        st.success(f"**Immediate Action**: Launch {len(launch_now_skus)} SKU{'s' if len(launch_now_skus) > 1 else ''} with full market rollout")
    
    if len(pilot_skus) > 0:
        st.warning(f"**Pilot Testing**: Test {len(pilot_skus)} SKU{'s' if len(pilot_skus) > 1 else ''} in 2-3 markets before full launch")
    
    if len(defer_skus) > 0:
        st.error(f"**Optimization Needed**: {len(defer_skus)} SKU{'s' if len(defer_skus) > 1 else ''} require performance improvement before launch")
    
    # Assumption note
    st.caption("📝 **Note**: Decisions based on composite score analysis combining Incremental Value, Capture Efficiency, Cannibalization, and Risk metrics. Confidence levels reflect model certainty.")

df = load_sku_data()

st.title("📊 SKU Portfolio Dashboard")
st.markdown("---")

if not df.empty:
    st.success(f"✅ Loaded {len(df)} SKUs successfully")
    
    # Add filterable views sidebar
    st.sidebar.header("🔍 Filter Options")
    st.sidebar.caption("Filter SKUs by different criteria to focus your analysis")
    
    # Initialize all filter states if not set
    if 'perspective' not in st.session_state:
        st.session_state.perspective = "Brand"
    if 'performance_tier' not in st.session_state:
        st.session_state.performance_tier = "All"
    if 'risk_level' not in st.session_state:
        st.session_state.risk_level = "All"
    if 'launch_readiness' not in st.session_state:
        st.session_state.launch_readiness = "All"
    # Note: analysis_perspective removed - now showing both Brand and Owner perspectives
    
    # Initialize active tab state to preserve tab selection during reruns
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 1  # Default to Overview tab (index 1)
    
    # Add clear filters button (before creating widgets)
    st.markdown("""
    <style>
    .stButton > button {
        padding: 0.5rem 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🔄 Clear All Filters"):
        # Reset all filter states to default values
        st.session_state.performance_tier = "All"
        st.session_state.risk_level = "All"
        st.session_state.launch_readiness = "All"
        st.session_state.perspective = "Brand"
        # Note: Removed st.rerun() to prevent tab reset
        st.sidebar.success("✅ Filters cleared!")
    
    # SKU Level Analysis - showing both Brand and Owner perspectives
    st.sidebar.info("📊 **SKU Level Analysis**: Showing both Brand and Owner perspectives")
    
    # Performance tier filter
    performance_tier = st.sidebar.selectbox(
        "📈 Performance Tier:",
        ["All", "High Performers", "Medium Performers", "Low Performers"],
        key="performance_tier",
        index=0,
        help="Filter by composite score performance"
    )
    
    # Risk level filter
    risk_level = st.sidebar.selectbox(
        "⚠️ Risk Level:",
        ["All", "Low Risk", "Medium Risk", "High Risk"],
        key="risk_level",
        index=0,
        help="Filter by VaR10 risk assessment"
    )
    
    # Launch readiness filter
    launch_readiness = st.sidebar.selectbox(
        "🚀 Launch Readiness:",
        ["All", "Ready to Launch", "Pilot Recommended", "Defer Launch"],
        key="launch_readiness",
        index=0,
        help="Filter by launch recommendation"
    )
    
    # Add bookmarked SKUs panel
    create_bookmarked_skus_panel()
    
    # Apply filters to dataframe
    filtered_df = df.copy()
    
    # Apply performance tier filter
    if performance_tier == "High Performers":
        filtered_df = filtered_df[filtered_df['Composite_Score'] >= filtered_df['Composite_Score'].quantile(0.75)]
    elif performance_tier == "Medium Performers":
        filtered_df = filtered_df[(filtered_df['Composite_Score'] >= filtered_df['Composite_Score'].quantile(0.25)) & 
                                 (filtered_df['Composite_Score'] < filtered_df['Composite_Score'].quantile(0.75))]
    elif performance_tier == "Low Performers":
        filtered_df = filtered_df[filtered_df['Composite_Score'] < filtered_df['Composite_Score'].quantile(0.25)]
    
    # Apply risk level filter
    if risk_level == "Low Risk":
        filtered_df = filtered_df[filtered_df['Brand_VaR10'] <= 4.0]
    elif risk_level == "Medium Risk":
        filtered_df = filtered_df[(filtered_df['Brand_VaR10'] > 4.0) & (filtered_df['Brand_VaR10'] <= 6.0)]
    elif risk_level == "High Risk":
        filtered_df = filtered_df[filtered_df['Brand_VaR10'] > 6.0]
    
    # Apply launch readiness filter
    if launch_readiness != "All":
        launch_ready_skus = []
        pilot_skus = []
        defer_skus = []
        
        for _, row in df.iterrows():
            recommendation, confidence, confidence_score, rec_color = get_action_recommendation(row)
            if "Launch" in recommendation and "confidence" in recommendation:
                launch_ready_skus.append(row['SKU'])
            elif "Pilot" in recommendation:
                pilot_skus.append(row['SKU'])
            else:
                defer_skus.append(row['SKU'])
        
        if launch_readiness == "Ready to Launch":
            filtered_df = filtered_df[filtered_df['SKU'].isin(launch_ready_skus)]
        elif launch_readiness == "Pilot Recommended":
            filtered_df = filtered_df[filtered_df['SKU'].isin(pilot_skus)]
        elif launch_readiness == "Defer Launch":
            filtered_df = filtered_df[filtered_df['SKU'].isin(defer_skus)]
    
    # Show filter results
    if len(filtered_df) != len(df):
        st.info(f"🔍 **Filtered Results**: Showing {len(filtered_df)} of {len(df)} SKUs")
        
        # Show active filters
        active_filters = []
        # Note: analysis_perspective removed - now showing both Brand and Owner perspectives
        if performance_tier != "All":
            active_filters.append(f"Performance: {performance_tier}")
        if risk_level != "All":
            active_filters.append(f"Risk: {risk_level}")
        if launch_readiness != "All":
            active_filters.append(f"Readiness: {launch_readiness}")
        
        if active_filters:
            st.caption(f"**Active Filters**: {', '.join(active_filters)}")
    else:
        st.caption("📊 **Showing all SKUs** - no filters applied")
    
    # Use filtered dataframe for all tabs
    df_to_use = filtered_df
    
    # Create tabs with shorter names to fit all in one row
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "How to use the dashboard", "📈 Overview", "🗺️ Risk", "⚠️ Cannibal", "📈 Lift", "📊 Multiples", 
        "🎯 Rivals", "📊 Compact", "📋 Executive", "🔍 Details", "📋 Reference", "🎯 Decision"
    ])
    
    with tab1:
        # Call the visual guide function (has its own header)
        create_visual_guide_appendix()
    
    with tab2:
        st.header("📈 Portfolio Leaderboard - SKU Level Analysis")
        st.markdown("*Showing both Brand and Owner perspectives for each SKU - sorted by composite score*")
        
        # Top 5 Launch Callout
        create_top5_launch_callout(df_to_use)
        # Layout selector
        layout_option = st.radio(
            "Choose Layout:",
            ["📄 Full Detail View", "🔲 4-Up Grid View"],
            horizontal=True
        )
        
        if layout_option == "🔲 4-Up Grid View":
            # 4-up grid layout
            st.subheader("🔲 Compact Comparison View")
            
            # Create rows of 4 SKUs each
            for i in range(0, len(df_to_use), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < len(df_to_use):
                        row = df_to_use.iloc[i + j]
                        with col:
                            create_compact_sku_card(row, i + j + 1)

        else:
            # Full detail view (existing code)
            for idx, row in df_to_use.iterrows():
                with st.container():
                    st.markdown("---", help="SKU separator")
                    # Add thicker separator
                    st.markdown("<hr style='border: 3px solid #333; margin: 20px 0;'>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 7])
                    
                    with col1:
                        # Top banner with SKU info and bookmark
                        col1_1, col1_2 = st.columns([3, 1])
                        with col1_1:
                            st.markdown(create_sku_banner(row), unsafe_allow_html=True)
                        with col1_2:
                            # Add some padding to prevent star from touching the edge
                            st.markdown("<div style='padding-left: 10px;'>", unsafe_allow_html=True)
                            is_bookmarked = create_bookmark_button(row['SKU'], row)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Action recommendation at the top (plain English)
                        plain_action = get_plain_english_action(row)
                        recommendation, confidence, confidence_score, rec_color = get_action_recommendation(row)
                        
                        if rec_color == "success":
                            st.success(plain_action)
                        elif rec_color == "warning":
                            st.warning(plain_action)
                        else:
                            st.error(plain_action)
                        
                        st.caption(f"*Confidence: {confidence} ({confidence_score:.0f}/100)*")
                        
                        st.markdown(f"**Rank:** #{idx + 1}")
                        st.markdown(f"**Composite Score:** {row['Composite_Score']:.1f}")
                        st.caption(get_assumption_note('composite_score'))
                        
                        risk_level, risk_color, risk_desc, risk_score = get_risk_tile(row)
                        if risk_color == "success":
                            st.success(f"**Risk:** {risk_level}")
                        elif risk_color == "warning":
                            st.warning(f"**Risk:** {risk_level}")
                        else:
                            st.error(f"**Risk:** {risk_level}")
                        st.caption(f"*{risk_desc}*")
                        
                        net_effect, net_color, net_desc = get_net_effect(row)
                        if net_color == "success":
                            st.success(f"**Owner Impact:** {net_effect}")
                        elif net_color == "warning":
                            st.warning(f"**Owner Impact:** {net_effect}")
                        else:
                            st.error(f"**Owner Impact:** {net_effect}")
                        st.caption(f"*{net_desc}*")
                    
                    with col2:
                        st.subheader("Key Metrics:")
                        
                        # Create 2x2 grid for metrics
                        metric_col1, metric_col2 = st.columns(2)
                        
                        with metric_col1:
                            # Dual perspective display for IV
                            brand_iv, owner_iv = get_dual_metric_display('IV', row)
                            st.markdown(f"**💰 Incremental Value:**")
                            st.markdown(f"📊 **Brand**: {brand_iv:.1f}% | 🏢 **Owner**: {owner_iv:.1f}%")
                            st.code(f"📊 Brand: {create_in_cell_bar(brand_iv, 8, 8)}")
                            st.code(f"🏢 Owner: {create_in_cell_bar(owner_iv, 8, 8)}")
                            st.caption(f"*Brand: {get_why_tag(brand_iv, 'IV')} | Owner: {get_why_tag(owner_iv, 'IV')}*")
                            
                            # Dual perspective display for Capture Efficiency
                            brand_capture, owner_capture = get_dual_metric_display('Capture_Eff', row)
                            st.markdown(f"**🎯 Capture Efficiency:**")
                            st.markdown(f"📊 **Brand**: {brand_capture:.1f}% | 🏢 **Owner**: {owner_capture:.1f}%")
                            st.code(f"📊 Brand: {create_in_cell_bar(brand_capture, 100, 8)}")
                            st.code(f"🏢 Owner: {create_in_cell_bar(owner_capture, 100, 8)}")
                            st.caption(f"*Brand: {get_why_tag(brand_capture, 'Capture')} | Owner: {get_why_tag(owner_capture, 'Capture')}*")
                        
                        with metric_col2:
                            # Dual perspective display for Cannibalization
                            brand_cannibal, owner_cannibal = get_dual_metric_display('Cannibal_Ratio', row)
                            st.markdown(f"**♻️ Cannibalization:**")
                            st.markdown(f"📊 **Brand**: {brand_cannibal:.1f}% | 🏢 **Owner**: {owner_cannibal:.1f}%")
                            st.code(f"📊 Brand: {create_in_cell_bar(brand_cannibal, 30, 8, reverse=True)}")
                            st.code(f"🏢 Owner: {create_in_cell_bar(owner_cannibal, 30, 8, reverse=True)}")
                            st.caption(f"*Brand: {get_why_tag(brand_cannibal, 'Cannibal')} | Owner: {get_why_tag(owner_cannibal, 'Cannibal')}*")
                            
                            # Dual perspective display for VaR10 Risk
                            brand_var, owner_var = get_dual_metric_display('VaR10', row)
                            st.markdown(f"**🛡️ VaR10 Risk:**")
                            st.markdown(f"📊 **Brand**: {brand_var:.1f}% | 🏢 **Owner**: {owner_var:.1f}%")
                            st.code(f"📊 Brand: {create_in_cell_bar(brand_var, 8, 8, reverse=True)}")
                            st.code(f"🏢 Owner: {create_in_cell_bar(owner_var, 8, 8, reverse=True)}")
                            st.caption(f"*Brand: {get_why_tag(brand_var, 'VaR')} | Owner: {get_why_tag(owner_var, 'VaR')}*")
                    
                        # Additional Metrics section
                        st.subheader("Additional Metrics:")
                        
                        # Create 2x2 grid for additional metrics
                        add_metric_col1, add_metric_col2 = st.columns(2)
                        
                        with add_metric_col1:
                            # Dual perspective display for Category Lift
                            brand_lift, owner_lift = get_dual_metric_display('Category_Lift', row)
                            st.markdown(f"**🌊 Category Lift:**")
                            st.markdown(f"📊 **Brand**: {brand_lift:.1f}% | 🏢 **Owner**: {owner_lift:.1f}%")
                            st.code(f"📊 Brand: {create_in_cell_bar(brand_lift, 8, 8)}")
                            st.code(f"🏢 Owner: {create_in_cell_bar(owner_lift, 8, 8)}")
                            st.caption(f"*Brand: {get_why_tag(brand_lift, 'Category_Lift')} | Owner: {get_why_tag(owner_lift, 'Category_Lift')}*")
                            
                            # Market HHI (same for both perspectives)
                            st.markdown(f"**🏪 Market HHI: {row['Market_HHI']:.1f}%**")
                            st.markdown(f"*{get_plain_english_headline('Market_HHI', row['Market_HHI'])}*")
                            st.code(create_in_cell_bar(row['Market_HHI'], 10, 8))
                            st.caption(f"*{get_why_tag(row['Market_HHI'], 'HHI')}*")
                        
                        with add_metric_col2:
                            # Dual perspective display for Confidence
                            brand_prob, owner_prob = get_dual_metric_display('IV_Prob', row)
                            st.markdown(f"**📊 Confidence:**")
                            st.markdown(f"📊 **Brand**: {brand_prob:.1f}% | 🏢 **Owner**: {owner_prob:.1f}%")
                            st.code(f"📊 Brand: {create_in_cell_bar(brand_prob, 100, 8)}")
                            st.code(f"🏢 Owner: {create_in_cell_bar(owner_prob, 100, 8)}")
                            st.caption(f"*Brand: {get_why_tag(brand_prob, 'Prob')} | Owner: {get_why_tag(owner_prob, 'Prob')}*")
                            
                            create_competitor_panel(row)
                        
                        # Summary section
                        st.subheader("Summary:")
                        st.info(f"{get_plain_english_summary(row)}\n\n**Technical:** {get_human_summary(row)}")
                        
                        # Add What Could Change box
                        create_what_could_change_box(row)
                    
                    st.markdown("**Portfolio Position Analysis:**")
                    scatter_chart = create_mini_scatter_plot(df_to_use, row['SKU'])
                    st.plotly_chart(scatter_chart, use_container_width=True)
                    st.caption(get_assumption_note('mini_scatter'))
                    
                    st.markdown("**Source of Volume Analysis:**")
                    chart = create_source_volume_chart(row)
                    st.plotly_chart(chart, use_container_width=True)
                    st.caption(get_assumption_note('source_volume'))
                    
                    
                    # Add Portfolio Ranking footer
                    create_sku_ranking_footer(row, df_to_use)
        
        # Add KPI strip only to Overview tab
        create_kpi_strip(df_to_use)
    
    with tab3:
        st.header(f"🗺️ Portfolio Risk Map {get_perspective_label()}")
        st.markdown("*Interactive risk analysis: VaR10 vs Incremental Value trade-offs*")
        
        # Risk map explanation
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🟢 **Green Zone**: Low Risk, High Return - Launch with confidence")
        with col2:
            st.warning("🟠 **Orange Zone**: Medium Risk - Monitor closely")
        with col3:
            st.error("🔴 **Red Zone**: High Risk - Consider pilot testing")
        
        # Create and display risk map
        risk_map = create_portfolio_risk_map(df_to_use)
        st.plotly_chart(risk_map, use_container_width=True)
        
        # Add assumption note
        st.caption(get_assumption_note('risk_map'))
        
        # Risk analysis summary
        st.subheader("📊 Risk Analysis Summary")
        
        # Calculate risk distribution
        low_risk = len(df_to_use[(df_to_use['Brand_VaR10'] <= 4) & (df_to_use['Brand_IV'] >= 5)])
        medium_risk = len(df_to_use[(df_to_use['Brand_VaR10'] > 4) & (df_to_use['Brand_VaR10'] <= 6)])
        high_risk = len(df_to_use[df_to_use['Brand_VaR10'] > 6])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total SKUs", len(df_to_use))
        with col2:
            delta_pct = f"{low_risk/len(df_to_use)*100:.0f}%" if len(df_to_use) > 0 else "0%"
            st.metric("Low Risk", low_risk, delta=delta_pct)
        with col3:
            delta_pct = f"{medium_risk/len(df_to_use)*100:.0f}%" if len(df_to_use) > 0 else "0%"
            st.metric("Medium Risk", medium_risk, delta=delta_pct)
        with col4:
            delta_pct = f"{high_risk/len(df_to_use)*100:.0f}%" if len(df_to_use) > 0 else "0%"
            st.metric("High Risk", high_risk, delta=delta_pct)
    
    with tab4:
        # Call the cannibalization watchlist function (has its own header)
        create_cannibalization_watchlist(df_to_use)
    
    with tab5:
        # Call the category lift table function (has its own header)
        create_category_lift_table(df_to_use)
    
    with tab6:
        # Call the small multiples function (has its own header)
        create_small_multiples(df_to_use)
    
    with tab7:
        # Call the closest rivals widget function (has its own header)
        create_closest_rivals_widget(df_to_use)
    
    with tab8:
        st.header("📊 Compact View")
        st.markdown("*Space-efficient view with in-cell bars and short labels*")
        
        # Call the compact summary cards function
        create_compact_summary_cards(df_to_use)
        
        st.markdown("---")
        
        # Call the compact metrics table function
        create_compact_metrics_table(df_to_use)
    
    with tab9:
        # Call the executive booklet function (has its own header)
        create_executive_booklet(df_to_use)
    
    with tab10:
        st.header("🔍 Complete Data View")
        st.markdown(create_hierarchical_table(df_to_use), unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📈 Portfolio Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Brand IV", f"{df_to_use['Brand_IV'].mean():.1f}%")
        with col2:
            st.metric("Avg Capture Eff", f"{df_to_use['Brand_Capture_Eff'].mean():.1f}%")
        with col3:
            st.metric("Avg Cannibal Ratio", f"{df_to_use['Brand_Cannibal_Ratio'].mean():.1f}%")
        with col4:
            st.metric("Avg Market HHI", f"{df_to_use['Market_HHI'].mean():.2f}%")
    
    with tab11:
        # Call the definitions and thresholds function (has its own header)
        create_definitions_thresholds_sheet()
    
    with tab12:
        # Call the simple decision page function (has its own header)
        create_simple_decision_page(df_to_use)
    
    # Add KPI strip only to Overview tab
    # create_kpi_strip(df_to_use)  # Moved to Overview tab only
    
else:
    st.error("❌ No data loaded")