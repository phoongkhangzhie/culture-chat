"""
Cultural taxonomy based on the CultureScope dimensional schema.

Each entry is a fine-grained, queryable dimension derived from the
CultureScope fine-grained dimension list.

Structure: Layer > Category > Topic Aspect > Fine-grained Dimension
"""

from src.models import CultureDimension

CULTURAL_DIMENSIONS: dict[str, CultureDimension] = {

    # =========================================================================
    # Layer 1: Institutional Norms
    # Category: Geography & Customs
    # Topic Aspect: Population and Geography
    # =========================================================================

    "population_rank": CultureDimension(
        name="Population Rank",
        category="Institutional Norms > Geography & Customs > Population and Geography",
        description="A nation's position by population size globally, shaping resource allocation, migration patterns, and international influence",
    ),

    "population_distribution": CultureDimension(
        name="Population Distribution",
        category="Institutional Norms > Geography & Customs > Population and Geography",
        description="How a country's population is spread across urban, rural, and regional areas, influencing cultural diversity and local customs",
    ),

    "land_area_percentage": CultureDimension(
        name="Land Area and Geography",
        category="Institutional Norms > Geography & Customs > Population and Geography",
        description="Proportion of different land types (urban, agricultural, forested) shaping environmental culture and land-use practices",
    ),

    "main_regions": CultureDimension(
        name="Main Regions",
        category="Institutional Norms > Geography & Customs > Population and Geography",
        description="Key geographic and administrative regions within a country and their distinct cultural identities and sub-cultures",
    ),

    "ethnicity": CultureDimension(
        name="Ethnicity and Racial Composition",
        category="Institutional Norms > Geography & Customs > Population and Geography",
        description="Ethnic and racial composition of a society and how diversity shapes national identity and intercultural dynamics",
    ),

    "official_languages": CultureDimension(
        name="Official Languages",
        category="Institutional Norms > Geography & Customs > Population and Geography",
        description="State-recognized languages used in government, education, and public life",
    ),

    "widely_spoken_languages": CultureDimension(
        name="Widely Spoken Languages",
        category="Institutional Norms > Geography & Customs > Population and Geography",
        description="Languages in common everyday use beyond official status, including regional dialects and lingua francas",
    ),

    "famous_rivers": CultureDimension(
        name="Famous Rivers and Waterways",
        category="Institutional Norms > Geography & Customs > Population and Geography",
        description="Notable rivers and waterways as cultural, historical, and geographic landmarks that shape regional identity",
    ),

    "climate_geography": CultureDimension(
        name="Climate and Physical Geography",
        category="Institutional Norms > Geography & Customs > Population and Geography",
        description="Climate zones and geographic features and their influence on lifestyle, agriculture, clothing, and cultural practices",
    ),

    # Topic Aspect: Dates of Significance
    # =========================================================================

    "national_holidays": CultureDimension(
        name="National Holidays",
        category="Institutional Norms > Geography & Customs > Dates of Significance",
        description="State-designated public holidays commemorating historical, political, or civic milestones",
    ),

    "religious_holidays": CultureDimension(
        name="Religious Holidays",
        category="Institutional Norms > Geography & Customs > Dates of Significance",
        description="Faith-based celebrations and observances that shape community life and the annual calendar",
    ),

    "cultural_holidays": CultureDimension(
        name="Cultural Holidays and Festivals",
        category="Institutional Norms > Geography & Customs > Dates of Significance",
        description="Non-religious and non-state festivals celebrating cultural heritage, seasonal cycles, or community traditions",
    ),

    "festival_origins": CultureDimension(
        name="Origin of Festivals",
        category="Institutional Norms > Geography & Customs > Dates of Significance",
        description="Historical, mythological, and religious backstories that give festivals their meaning and shape how they are observed",
    ),

    "festival_celebrations": CultureDimension(
        name="Festival Celebration Practices",
        category="Institutional Norms > Geography & Customs > Dates of Significance",
        description="Rituals, activities, foods, and communal behaviors associated with celebrating key festivals",
    ),

    "festival_symbols": CultureDimension(
        name="Festival Symbols",
        category="Institutional Norms > Geography & Customs > Dates of Significance",
        description="Visual, material, and symbolic elements—such as colors, objects, and icons—associated with specific festivals",
    ),

    # =========================================================================
    # Layer 1: Institutional Norms
    # Category: Regulation & Policy
    # Topic Aspect: Transportation Rules
    # =========================================================================

    "vehicle_movement_rules": CultureDimension(
        name="Vehicle Movement Rules",
        category="Institutional Norms > Regulation & Policy > Transportation Rules",
        description="Country-specific regulations governing which side of the road vehicles travel on and general traffic flow conventions",
    ),

    "traffic_signs_and_signals": CultureDimension(
        name="Traffic Signs and Signals",
        category="Institutional Norms > Regulation & Policy > Transportation Rules",
        description="The system of road signs, traffic lights, and visual communication used to regulate vehicle and pedestrian movement",
    ),

    "pedestrian_rules": CultureDimension(
        name="Pedestrian and Non-Motorized Vehicle Rules",
        category="Institutional Norms > Regulation & Policy > Transportation Rules",
        description="Rights and responsibilities of pedestrians, cyclists, and non-motorized road users in traffic",
    ),

    "motor_vehicle_driving_rules": CultureDimension(
        name="Motor Vehicle Driving Rules",
        category="Institutional Norms > Regulation & Policy > Transportation Rules",
        description="Licensing requirements, speed limits, and behavioral standards for operating motor vehicles",
    ),

    "parking_regulations": CultureDimension(
        name="Parking Regulations",
        category="Institutional Norms > Regulation & Policy > Transportation Rules",
        description="Rules and norms governing where and how vehicles may be parked in public and private spaces",
    ),

    # Topic Aspect: Data Format
    # =========================================================================

    "date_order": CultureDimension(
        name="Date Order Convention",
        category="Institutional Norms > Regulation & Policy > Data Format",
        description="The sequence of day, month, and year components in written date representations (e.g., DD/MM/YYYY vs MM/DD/YYYY)",
    ),

    "date_separator": CultureDimension(
        name="Date and Number Separator",
        category="Institutional Norms > Regulation & Policy > Data Format",
        description="Characters used to separate date or number components, such as slashes, hyphens, periods, or commas",
    ),

    "year_format": CultureDimension(
        name="Year Format",
        category="Institutional Norms > Regulation & Policy > Data Format",
        description="Whether years are written in full four-digit form or abbreviated two-digit form in different contexts",
    ),

    "month_representation": CultureDimension(
        name="Month Representation",
        category="Institutional Norms > Regulation & Policy > Data Format",
        description="Whether months are written numerically, as abbreviations, or spelled out in full",
    ),

    "zero_padding": CultureDimension(
        name="Zero-Padding in Dates",
        category="Institutional Norms > Regulation & Policy > Data Format",
        description="The convention of adding leading zeros to single-digit day or month values in date notation",
    ),

    "natural_language_date": CultureDimension(
        name="Natural Language Date Expression",
        category="Institutional Norms > Regulation & Policy > Data Format",
        description="How dates are expressed conversationally in speech, including ordinal usage and word order",
    ),

    "calendar_system": CultureDimension(
        name="Calendar System",
        category="Institutional Norms > Regulation & Policy > Data Format",
        description="The primary calendar in use—Gregorian, lunar, lunisolar, or other—and how it governs daily and civic life",
    ),

    # Topic Aspect: Measurement Unit
    # =========================================================================

    "measurement_system": CultureDimension(
        name="Measurement System",
        category="Institutional Norms > Regulation & Policy > Measurement Unit",
        description="Whether a country uses the metric, imperial, or a traditional measurement system for everyday and official purposes",
    ),

    "unit_localization": CultureDimension(
        name="Unit Localization",
        category="Institutional Norms > Regulation & Policy > Measurement Unit",
        description="How measurement units are adapted in localized contexts such as food, medicine, clothing sizes, and construction",
    ),

    # Topic Aspect: Financial Market Rules
    # =========================================================================

    "financial_regulation_structure": CultureDimension(
        name="Financial Regulation Structure",
        category="Institutional Norms > Regulation & Policy > Financial Market Rules",
        description="The institutional framework for financial oversight, including central banks, regulatory bodies, and supervisory authorities",
    ),

    "financial_market_type": CultureDimension(
        name="Financial Market Type",
        category="Institutional Norms > Regulation & Policy > Financial Market Rules",
        description="Whether the financial system is bank-based, market-based, or a hybrid, and the role of public versus private ownership",
    ),

    "financial_entity_types": CultureDimension(
        name="Financial Entity Types",
        category="Institutional Norms > Regulation & Policy > Financial Market Rules",
        description="Categories of financial institutions operating in the country, such as commercial banks, cooperatives, and microfinance institutions",
    ),

    "financial_access_licensing": CultureDimension(
        name="Financial Access and Licensing",
        category="Institutional Norms > Regulation & Policy > Financial Market Rules",
        description="Requirements for obtaining licenses and market access for domestic and foreign financial entities",
    ),

    "financial_conduct_compliance": CultureDimension(
        name="Financial Conduct and Compliance",
        category="Institutional Norms > Regulation & Policy > Financial Market Rules",
        description="Rules governing how financial entities must behave, including anti-money laundering requirements and consumer protection standards",
    ),

    "financial_capital_risk": CultureDimension(
        name="Capital Requirements and Risk Management",
        category="Institutional Norms > Regulation & Policy > Financial Market Rules",
        description="Capital adequacy requirements and risk management frameworks that financial institutions must maintain",
    ),

    "monetary_payment_system": CultureDimension(
        name="Monetary and Payment System",
        category="Institutional Norms > Regulation & Policy > Financial Market Rules",
        description="The monetary policy framework, currency regime, and payment infrastructure including cash, card, and digital payment norms",
    ),

    "fintech_regulation": CultureDimension(
        name="FinTech Regulation",
        category="Institutional Norms > Regulation & Policy > Financial Market Rules",
        description="Regulatory treatment of financial technology companies, digital assets, and innovation sandboxes",
    ),

    "financial_tax_accounting": CultureDimension(
        name="Financial Tax and Accounting Standards",
        category="Institutional Norms > Regulation & Policy > Financial Market Rules",
        description="Taxation of financial transactions and the accounting standards used by financial institutions",
    ),

    "financial_international_alignment": CultureDimension(
        name="International Financial Alignment",
        category="Institutional Norms > Regulation & Policy > Financial Market Rules",
        description="Degree to which the financial system adheres to international standards such as Basel III, IFRS, and FATF recommendations",
    ),

    "financial_stability_resolution": CultureDimension(
        name="Financial Stability and Resolution",
        category="Institutional Norms > Regulation & Policy > Financial Market Rules",
        description="Frameworks for handling financial crises, bank failures, and systemic risk to protect depositors and the economy",
    ),

    # =========================================================================
    # Layer 2: Behavioral Patterns
    # Category: Personal Choices & Habits
    # Topic Aspect: Daily Life and Travel
    # =========================================================================

    "payment_habits": CultureDimension(
        name="Payment Habits",
        category="Behavioral Patterns > Personal Choices & Habits > Daily Life and Travel",
        description="Preferred payment methods in everyday transactions, including attitudes toward cash, card, and mobile payment",
    ),

    "travel_habits": CultureDimension(
        name="Travel Habits",
        category="Behavioral Patterns > Personal Choices & Habits > Daily Life and Travel",
        description="Patterns of domestic and international travel, commuting preferences, and common modes of transportation",
    ),

    "bathing_habits": CultureDimension(
        name="Bathing and Hygiene Habits",
        category="Behavioral Patterns > Personal Choices & Habits > Daily Life and Travel",
        description="Cultural norms around bathing frequency, timing, bathing facilities, and personal hygiene routines",
    ),

    "pet_raising_habits": CultureDimension(
        name="Pet-Raising Habits",
        category="Behavioral Patterns > Personal Choices & Habits > Daily Life and Travel",
        description="Cultural attitudes toward keeping pets, preferred animal companions, and norms around pet care and ownership",
    ),

    "social_media_usage": CultureDimension(
        name="Social Media Usage Patterns",
        category="Behavioral Patterns > Personal Choices & Habits > Daily Life and Travel",
        description="Dominant social media platforms, usage frequency, and the role of social media in daily social interaction and communication",
    ),

    "marriage_customs": CultureDimension(
        name="Marriage Customs",
        category="Behavioral Patterns > Personal Choices & Habits > Daily Life and Travel",
        description="Traditions, rituals, and social expectations surrounding courtship, engagement, wedding ceremonies, and marriage",
    ),

    # Topic Aspect: Diet and Health Concept
    # =========================================================================

    "eating_habits": CultureDimension(
        name="Eating Habits",
        category="Behavioral Patterns > Personal Choices & Habits > Diet and Health Concept",
        description="Typical meal patterns, dietary staples, food preferences, and the social and ceremonial role of eating",
    ),

    "attitudes_lifecycle": CultureDimension(
        name="Attitudes Toward Birth, Aging, Illness, and Death",
        category="Behavioral Patterns > Personal Choices & Habits > Diet and Health Concept",
        description="Cultural beliefs, rituals, and practices surrounding the major life events of birth, aging, illness, and death",
    ),

    "traditional_vs_modern_medicine": CultureDimension(
        name="Traditional vs. Modern Medicine",
        category="Behavioral Patterns > Personal Choices & Habits > Diet and Health Concept",
        description="The balance between traditional, herbal, and folk medicine and modern biomedical healthcare in everyday health decisions",
    ),

    # Topic Aspect: Education and Knowledge
    # =========================================================================

    "educational_practices": CultureDimension(
        name="Educational Practices",
        category="Behavioral Patterns > Personal Choices & Habits > Education and Knowledge",
        description="Teaching methodologies, classroom dynamics, and the structural features of the educational system",
    ),

    "teacher_student_relationship": CultureDimension(
        name="Teacher-Student Relationship",
        category="Behavioral Patterns > Personal Choices & Habits > Education and Knowledge",
        description="Cultural expectations and power dynamics in the relationship between educators and students",
    ),

    "academic_major_selection": CultureDimension(
        name="Academic Major Selection",
        category="Behavioral Patterns > Personal Choices & Habits > Education and Knowledge",
        description="Social, familial, and economic influences on students' choices of academic fields and areas of study",
    ),

    "career_choices": CultureDimension(
        name="Career Choices and Aspirations",
        category="Behavioral Patterns > Personal Choices & Habits > Education and Knowledge",
        description="Cultural attitudes toward career paths, including prestige hierarchies, family expectations, and individual autonomy",
    ),

    # Topic Aspect: Art and Entertainment
    # =========================================================================

    "diaspora_festive_activities": CultureDimension(
        name="Diaspora Festive Activities",
        category="Behavioral Patterns > Personal Choices & Habits > Art and Entertainment",
        description="How communities with shared cultural origins celebrate festivals differently depending on their geographic location and diaspora context",
    ),

    "traditional_music_and_dance": CultureDimension(
        name="Traditional Music and Dance",
        category="Behavioral Patterns > Personal Choices & Habits > Art and Entertainment",
        description="Indigenous and traditional musical styles, instruments, and dance forms as expressions of cultural heritage",
    ),

    "contemporary_art": CultureDimension(
        name="Contemporary Art and Culture",
        category="Behavioral Patterns > Personal Choices & Habits > Art and Entertainment",
        description="Modern artistic expressions including visual arts, film, literature, and performance that reflect contemporary cultural identity",
    ),

    # Topic Aspect: Personal Etiquette
    # =========================================================================

    "basic_etiquette": CultureDimension(
        name="Basic Etiquette",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Foundational rules of polite social behavior and conduct in everyday interpersonal interactions",
    ),

    "naming_convention": CultureDimension(
        name="Naming Convention",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="The structure and order of personal names, including given names, family names, and the use of honorifics",
    ),

    "name_origin": CultureDimension(
        name="Name Origin",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Cultural, religious, and historical sources from which personal names are drawn",
    ),

    "name_significance": CultureDimension(
        name="Name Significance",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="The meaning, symbolism, and cultural importance assigned to personal names within a culture",
    ),

    "punctuality_visiting": CultureDimension(
        name="Punctuality When Visiting",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Cultural expectations around arriving on time, early, or fashionably late when visiting someone's home",
    ),

    "shoe_etiquette": CultureDimension(
        name="Shoe Etiquette During a Visit",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Norms about whether to remove shoes before entering a home or other indoor spaces when visiting",
    ),

    "guest_hospitality": CultureDimension(
        name="Hospitality Customs When Receiving Guests",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="The rituals, obligations, and expected behaviors associated with welcoming and hosting visitors in the home",
    ),

    "host_gift_customs": CultureDimension(
        name="Bringing Gifts for the Host",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Expectations around whether and what type of gift to bring when visiting someone's home",
    ),

    "seating_etiquette": CultureDimension(
        name="Seating Etiquette for Guests and Hosts",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Rules and customs governing where guests and hosts sit relative to each other in social settings",
    ),

    "serving_etiquette": CultureDimension(
        name="Serving Etiquette During a Visit",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Norms around offering and serving food and drink to guests, including order of service, refills, and insistence",
    ),

    "leaving_food_etiquette": CultureDimension(
        name="Leaving Food After Eating",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Whether leaving food on the plate is considered polite, impolite, or neutral in a cultural dining context",
    ),

    "salt_etiquette": CultureDimension(
        name="Using Salt While Eating",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Cultural attitudes toward adding salt to food at the table and what this communicates to the host about the meal",
    ),

    "meal_compliments": CultureDimension(
        name="Giving Compliments During a Meal",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Norms around complimenting the host's cooking during or after a meal and the culturally expected responses",
    ),

    "right_hand_eating": CultureDimension(
        name="Eating with the Right Hand",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Cultural significance of using the right hand for eating and the social taboos associated with using the left hand",
    ),

    "alcohol_etiquette": CultureDimension(
        name="Alcohol Etiquette",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Cultural norms around offering, accepting, or declining alcoholic beverages in social and dining settings",
    ),

    "pork_dietary_norms": CultureDimension(
        name="Pork and Dietary Taboos",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Religious, cultural, and personal attitudes toward consuming pork, including taboos and culturally accepted alternatives",
    ),

    "gift_giving_gesture": CultureDimension(
        name="Handing Gifts",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="The physical and ceremonial manner in which gifts are presented, including which hand to use and whether to bow",
    ),

    "gifts_for_children": CultureDimension(
        name="Gifts for Children",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Appropriate and inappropriate gift choices for children, and norms around giving gifts directly to minors",
    ),

    "gift_opening_norms": CultureDimension(
        name="Opening Gifts",
        category="Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
        description="Whether gifts are opened immediately in front of the giver or set aside to be opened privately later",
    ),

    # Topic Aspect: Etiquette and Courtesy
    # =========================================================================

    "general_greeting_principles": CultureDimension(
        name="General Greeting Principles",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Core cultural norms governing how people initiate and respond to greetings across different social contexts",
    ),

    "male_male_greetings": CultureDimension(
        name="Greetings Between Men",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Physical and verbal norms for greetings between men, including handshakes, hugs, cheek kisses, and bowing",
    ),

    "female_female_greetings": CultureDimension(
        name="Greetings Between Women",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Physical and verbal norms for greetings between women, including physical contact and terms of address",
    ),

    "male_female_greetings": CultureDimension(
        name="Greetings Between Men and Women",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Norms governing cross-gender greetings, including appropriate physical contact and verbal forms of address",
    ),

    "business_appointment_scheduling": CultureDimension(
        name="Business Appointment Scheduling",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Expectations around how far in advance meetings should be arranged and the etiquette of confirming appointments",
    ),

    "business_dress_code": CultureDimension(
        name="Business Dress Code",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Expected attire in professional and business settings, including formal, smart-casual, and industry-specific norms",
    ),

    "business_card_exchange": CultureDimension(
        name="Business Card Exchange",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Rituals and significance of exchanging business cards, including how to give and receive them respectfully",
    ),

    "business_network_building": CultureDimension(
        name="Business Network Building",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="The role of relationship-building and social connections that precede or enable formal business dealings",
    ),

    "seniority_in_business": CultureDimension(
        name="Age and Experience in Business",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="How age, seniority, and experience factor into business introductions, seating order, and protocol",
    ),

    "business_familiarity": CultureDimension(
        name="Familiarity Before Business Meetings",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Expectations around the level of personal familiarity and rapport required before transitioning to formal business",
    ),

    "business_socialization": CultureDimension(
        name="Socialization During Business Meetings",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="The extent to which small talk and personal interaction are expected as part of a formal business meeting",
    ),

    "meeting_duration": CultureDimension(
        name="Meeting Duration",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Cultural expectations around how long business meetings typically last and punctuality norms for starting and ending",
    ),

    "open_door_policy": CultureDimension(
        name="Open Door Policy in Business",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Whether unscheduled interruptions and walk-ins are acceptable practice during business meetings",
    ),

    "business_meeting_interruptions": CultureDimension(
        name="Interruptions During Business Meetings",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Cultural attitudes toward being interrupted or interrupting others during professional discussions",
    ),

    "deference_to_seniority": CultureDimension(
        name="Deference to Senior in Business",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Norms for showing respect to more senior or higher-ranking participants in a business meeting",
    ),

    "negotiation_style": CultureDimension(
        name="Negotiation Style",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Whether the dominant approach to business negotiation is direct or indirect, competitive or collaborative, quick or deliberate",
    ),

    "business_decision_making": CultureDimension(
        name="Business Decision-Making",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="How decisions are reached in business settings—individually by a leader, by group consensus, or through hierarchical approval",
    ),

    "business_bartering": CultureDimension(
        name="Bartering in Business",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="The role and acceptability of price negotiation and bartering in business transactions",
    ),

    "private_business_meetings": CultureDimension(
        name="Private Meetings During Negotiations",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Norms around requesting or holding side meetings and one-on-one discussions during broader group negotiations",
    ),

    "confrontation_avoidance_business": CultureDimension(
        name="Confrontation Avoidance in Business",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Cultural preferences for avoiding direct disagreement or open conflict in professional and business settings",
    ),

    "business_meeting_followup": CultureDimension(
        name="Business Meeting Follow-Up",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Expectations around post-meeting communication, such as follow-up emails, calls, written summaries, or courtesy gifts",
    ),

    "ongoing_negotiations": CultureDimension(
        name="Ongoing Negotiations After a Meeting",
        category="Behavioral Patterns > Personal Choices & Habits > Etiquette and Courtesy",
        description="Norms for continuing to negotiate terms after an initial agreement appears to have been reached",
    ),

    # Topic Aspect: Communication Style
    # =========================================================================

    "verbal_communication_style": CultureDimension(
        name="Verbal Communication Style",
        category="Behavioral Patterns > Personal Choices & Habits > Communication Style",
        description="The general directness or indirectness of verbal communication in social and professional contexts",
    ),

    "indirect_communication": CultureDimension(
        name="Indirect Communication",
        category="Behavioral Patterns > Personal Choices & Habits > Communication Style",
        description="The use of implication, metaphor, and non-literal language to convey meaning without explicit statement",
    ),

    "cultural_humour": CultureDimension(
        name="Cultural Humour",
        category="Behavioral Patterns > Personal Choices & Habits > Communication Style",
        description="The style, subjects, and social function of humor, including what is considered funny, offensive, or inappropriate",
    ),

    "physical_contact_norms": CultureDimension(
        name="Physical Contact Norms",
        category="Behavioral Patterns > Personal Choices & Habits > Communication Style",
        description="Norms around touching in social interaction, including acceptable forms of physical contact between acquaintances and strangers",
    ),

    "personal_space_norms": CultureDimension(
        name="Personal Space",
        category="Behavioral Patterns > Personal Choices & Habits > Communication Style",
        description="Cultural expectations around appropriate physical distance between people in conversation and in public spaces",
    ),

    "communication_gestures": CultureDimension(
        name="Gestures in Communication",
        category="Behavioral Patterns > Personal Choices & Habits > Communication Style",
        description="The meaning and use of hand gestures, facial expressions, and body language in everyday communication",
    ),

    "beckoning_gestures": CultureDimension(
        name="Beckoning Gestures",
        category="Behavioral Patterns > Personal Choices & Habits > Communication Style",
        description="Culturally specific hand or finger gestures used to call or beckon another person, which vary widely across cultures",
    ),

    "eye_contact_norms": CultureDimension(
        name="Eye Contact Norms",
        category="Behavioral Patterns > Personal Choices & Habits > Communication Style",
        description="Cultural norms around the appropriateness, meaning, and expected duration of direct eye contact in social and professional contexts",
    ),

    # Topic Aspect: Fixed Expressions in Language
    # =========================================================================

    "linguistic_idioms": CultureDimension(
        name="Linguistic Idioms",
        category="Behavioral Patterns > Personal Choices & Habits > Fixed Expressions in Language",
        description="Fixed phrases whose meaning cannot be inferred from the literal meaning of their component words",
    ),

    "common_sayings": CultureDimension(
        name="Common Sayings",
        category="Behavioral Patterns > Personal Choices & Habits > Fixed Expressions in Language",
        description="Widely used expressions that encapsulate cultural wisdom, social norms, or shared values in everyday speech",
    ),

    "proverbs": CultureDimension(
        name="Proverbs",
        category="Behavioral Patterns > Personal Choices & Habits > Fixed Expressions in Language",
        description="Traditional sayings that convey moral lessons or practical wisdom passed down through generations",
    ),

    "neologisms_and_abbreviations": CultureDimension(
        name="Neologisms and Abbreviations",
        category="Behavioral Patterns > Personal Choices & Habits > Fixed Expressions in Language",
        description="New words, slang, and abbreviations that have emerged in contemporary usage, often driven by technology and pop culture",
    ),

    # =========================================================================
    # Layer 3: Core Values and Social Structures
    # Category: Social Relationship and Structures
    # Topic Aspect: Family Dynamics
    # =========================================================================

    "communal_living": CultureDimension(
        name="Communal Living",
        category="Core Values and Social Structures > Social Relationship and Structures > Family Dynamics",
        description="Arrangements where multiple generations or extended family members share living spaces, resources, and responsibilities",
    ),

    "parental_care_norms": CultureDimension(
        name="Parental Care",
        category="Core Values and Social Structures > Social Relationship and Structures > Family Dynamics",
        description="Cultural expectations around who bears responsibility for childcare and how parental roles are defined and shared",
    ),

    # Topic Aspect: Household Structures
    # =========================================================================

    "patriarchal_structures": CultureDimension(
        name="Patriarchal Structures",
        category="Core Values and Social Structures > Social Relationship and Structures > Household Structures",
        description="The extent to which male authority dominates family, household, and community decision-making",
    ),

    "womens_family_roles": CultureDimension(
        name="Women's Roles in the Family",
        category="Core Values and Social Structures > Social Relationship and Structures > Household Structures",
        description="Cultural expectations placed on women regarding domestic responsibilities, work, education, and family leadership",
    ),

    "household_social_interaction": CultureDimension(
        name="Social Interaction Within Households",
        category="Core Values and Social Structures > Social Relationship and Structures > Household Structures",
        description="Patterns of socializing within and between households, including who is welcomed into the home and under what conditions",
    ),

    "urban_rural_divide": CultureDimension(
        name="Urban-Rural Divide",
        category="Core Values and Social Structures > Social Relationship and Structures > Household Structures",
        description="Differences in lifestyle, values, opportunities, and cultural norms between urban and rural populations",
    ),

    "seniority_and_childhood": CultureDimension(
        name="Seniority and Childhood",
        category="Core Values and Social Structures > Social Relationship and Structures > Household Structures",
        description="Cultural constructions of age-based hierarchy and the social boundaries between childhood and adulthood",
    ),

    # Topic Aspect: Gender Roles
    # =========================================================================

    "male_dominance": CultureDimension(
        name="Male Dominance",
        category="Core Values and Social Structures > Social Relationship and Structures > Gender Roles",
        description="Cultural norms and social structures that position men as the primary authority in family, professional, and public life",
    ),

    "gender_social_compliance": CultureDimension(
        name="Gender Social Compliance",
        category="Core Values and Social Structures > Social Relationship and Structures > Gender Roles",
        description="Social pressures to conform to culturally expected gender behaviors, roles, and presentations",
    ),

    "gender_and_honour": CultureDimension(
        name="Gender and Honour",
        category="Core Values and Social Structures > Social Relationship and Structures > Gender Roles",
        description="The intersection of gender identity with cultural concepts of honor, shame, and reputation in the community",
    ),

    "changing_gender_attitudes": CultureDimension(
        name="Changing Attitudes Toward Gender",
        category="Core Values and Social Structures > Social Relationship and Structures > Gender Roles",
        description="Evolving social norms and generational shifts in how gender roles, equality, and identity are understood",
    ),

    # =========================================================================
    # Layer 3: Core Values and Social Structures
    # Category: Values and Beliefs
    # Topic Aspect: Cultural Values
    # =========================================================================

    "power_distance": CultureDimension(
        name="Power Distance",
        category="Core Values and Social Structures > Values and Beliefs > Cultural Values",
        description="The degree to which less powerful members of a society accept and expect unequal distribution of power and authority",
    ),

    "collectivism": CultureDimension(
        name="Collectivism",
        category="Core Values and Social Structures > Values and Beliefs > Cultural Values",
        description="The cultural tendency to prioritize group goals, family loyalty, and social harmony over individual achievement and autonomy",
    ),

    "individualism": CultureDimension(
        name="Individualism",
        category="Core Values and Social Structures > Values and Beliefs > Cultural Values",
        description="The cultural emphasis on personal autonomy, self-reliance, privacy, and individual rights over group obligations",
    ),

    "achievement_motivation": CultureDimension(
        name="Motivation Toward Achievement and Success",
        category="Core Values and Social Structures > Values and Beliefs > Cultural Values",
        description="The degree to which a society values competition, assertiveness, material success, and performance",
    ),

    "uncertainty_avoidance": CultureDimension(
        name="Uncertainty Avoidance",
        category="Core Values and Social Structures > Values and Beliefs > Cultural Values",
        description="The extent to which a culture tolerates ambiguity and relies on formal rules or structured environments to cope with uncertainty",
    ),

    "long_term_orientation": CultureDimension(
        name="Long-Term Orientation",
        category="Core Values and Social Structures > Values and Beliefs > Cultural Values",
        description="The degree to which a society prioritizes future rewards, perseverance, and thrift versus short-term gains and tradition",
    ),

    "cultural_indulgence": CultureDimension(
        name="Indulgence",
        category="Core Values and Social Structures > Values and Beliefs > Cultural Values",
        description="The extent to which a culture allows gratification of basic human desires related to enjoying life, leisure, and having fun",
    ),

    # Topic Aspect: Religion
    # =========================================================================

    "religious_beliefs_and_practices": CultureDimension(
        name="Religious Beliefs and Practices",
        category="Core Values and Social Structures > Values and Beliefs > Religion",
        description="The role of religious belief systems in shaping individual identity, community life, moral frameworks, and daily behavior",
    ),

    # Topic Aspect: Do's and Don'ts
    # =========================================================================

    "modest_dress_norms": CultureDimension(
        name="Modest Dress Norms",
        category="Core Values and Social Structures > Values and Beliefs > Do's and Don'ts",
        description="Cultural expectations around covering the body, particularly in public spaces, religious sites, and formal settings",
    ),

    "informality_norms": CultureDimension(
        name="Informality Norms",
        category="Core Values and Social Structures > Values and Beliefs > Do's and Don'ts",
        description="Contexts in which informal speech, behavior, and dress are appropriate or considered disrespectful",
    ),

    "compliment_norms": CultureDimension(
        name="Compliment Norms",
        category="Core Values and Social Structures > Values and Beliefs > Do's and Don'ts",
        description="How and when to give and receive compliments appropriately, and how false modesty or direct acceptance is perceived",
    ),

    "cultural_acknowledgement": CultureDimension(
        name="Cultural Acknowledgement",
        category="Core Values and Social Structures > Values and Beliefs > Do's and Don'ts",
        description="The importance of recognizing and showing genuine respect for local customs, history, and cultural identity",
    ),

    "education_as_value": CultureDimension(
        name="Education as a Cultural Value",
        category="Core Values and Social Structures > Values and Beliefs > Do's and Don'ts",
        description="The cultural weight placed on formal education as a marker of social status, capability, and respectability",
    ),

    "cultural_insults": CultureDimension(
        name="Cultural Insults",
        category="Core Values and Social Structures > Values and Beliefs > Do's and Don'ts",
        description="Words, gestures, and behaviors that are considered deeply insulting or offensive within a specific cultural context",
    ),

    "inappropriate_humor": CultureDimension(
        name="Inappropriate Humor",
        category="Core Values and Social Structures > Values and Beliefs > Do's and Don'ts",
        description="Contexts and settings in which crude, sexual, or offensive humor is considered taboo or socially unacceptable",
    ),

    "political_criticism_norms": CultureDimension(
        name="Political Criticism Norms",
        category="Core Values and Social Structures > Values and Beliefs > Do's and Don'ts",
        description="The social and legal boundaries around publicly criticizing political leaders, government policies, or the state",
    ),

    "culturally_sensitive_topics": CultureDimension(
        name="Culturally Sensitive Topics",
        category="Core Values and Social Structures > Values and Beliefs > Do's and Don'ts",
        description="Subjects that are considered taboo, divisive, or inappropriate to raise in casual social conversation",
    ),

    "ethnicity_assumptions": CultureDimension(
        name="Ethnicity Assumptions",
        category="Core Values and Social Structures > Values and Beliefs > Do's and Don'ts",
        description="The social pitfalls of making assumptions about a person's background, nationality, or identity based on appearance",
    ),

    "cultural_stereotyping": CultureDimension(
        name="Cultural Stereotyping",
        category="Core Values and Social Structures > Values and Beliefs > Do's and Don'ts",
        description="The harm caused by applying overgeneralized cultural assumptions to individuals based on their group membership",
    ),
}


def get_dimension(key: str) -> CultureDimension:
    """Get a CultureDimension by its key."""
    if key not in CULTURAL_DIMENSIONS:
        available = list(CULTURAL_DIMENSIONS.keys())
        raise ValueError(
            f"Unknown dimension '{key}'.\nAvailable dimensions: {available}"
        )
    return CULTURAL_DIMENSIONS[key]


def list_dimensions() -> dict[str, str]:
    """Return a mapping of dimension key → description."""
    return {k: v.description for k, v in CULTURAL_DIMENSIONS.items()}
