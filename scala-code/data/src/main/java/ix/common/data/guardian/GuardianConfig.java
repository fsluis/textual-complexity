package ix.common.data.guardian;

import ix.data.hadoop.HadoopRecord;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 1/31/12
 * Time: 5:25 PM
 * To change this template use File | Settings | File Templates.
 */
public class GuardianConfig {
    public static class GuardianRecord extends HadoopRecord {
        public static final Map<String, Integer> TYPES = new LinkedHashMap<String, Integer>() {{
            put("id", LONG);
            put("content", STRING);
        }};

        @Override
        public Map<String, Integer> getTypes() {
            return TYPES;
        }
    }
}
