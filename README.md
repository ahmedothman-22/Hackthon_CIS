## 🍽️ Food Waste Prediction Page  

This page is a core feature of **GivEat**, where users (e.g., restaurants, event organizers) can input details about their food and predict possible waste.  

![Food Waste Prediction Demo]("Screenshot%202025-08-30.png)

### 🔑 How it Works:  
1. **Inputs**:  
   - *Number of Guests*: Total number of expected guests.  
   - *Type of Food*: Select from categories (e.g., Meat, Vegetables, etc.).  
   - *Per Guest Quantity (kg)*: Estimated food per guest.  
   - *Perishability Score (1-2)*: Rate how quickly the food spoils.  
   - *Is Buffet*: Yes/No toggle to mark if it’s a buffet setting.  
   - *Cold Chain Flag*: Yes/No to indicate if cold chain preservation exists.  
   - *Geographical Location*: Location where the food is prepared.  

2. **Output**:  
   - **Will Waste**: Prediction if surplus food will be generated.  
   - **Probability**: Likelihood of surplus food.  
   - **Matching Charity**: Suggests the nearest charity/food bank to receive the surplus, with contact info and required quantity.  
   - **Advice**: Smart recommendations (e.g., store proteins in freezer, cut large portions into smaller meals, prepare smaller quantities first).  

✨ This feature helps donors **predict, reduce, and redirect** surplus food before it goes to waste.
