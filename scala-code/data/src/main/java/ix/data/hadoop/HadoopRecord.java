package ix.data.hadoop;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.lib.db.DBWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.math.BigInteger;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

import static org.apache.commons.lang.StringEscapeUtils.escapeJava;
import static org.apache.commons.lang.StringEscapeUtils.unescapeJava;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 1/28/11
 * Time: 4:03 PM
 * To change this template use File | Settings | File Templates.
 */
public abstract class HadoopRecord extends HashMap<String,Object> implements Writable, DBWritable {
    public static final int
        BOOL=1,BYTE=2,CHAR=3,DOUBLE=4,FLOAT=5,INT=6,LONG=7,SHORT=8,STRING=9,TEXT=10;
    public static boolean safeMode =true;

    public HadoopRecord() {
        super();
        //for(String key : getTypes().keySet())
        //    put(key, null);
    }

    public void write(DataOutput out) throws IOException {
        for(Map.Entry<String,Integer> type : getTypes().entrySet()) {
            String key = type.getKey();
            Object value = get(key);

            //Check for NaNs
            if(value!=null) {
                if(type.getValue()==DOUBLE)
                    if (Double.isNaN((Double)value))
                        value = null;
                if(type.getValue()==FLOAT)
                    if (Float.isNaN((Float)value))
                        value = null;
            }

            //Write stuff
            if(value==null) {
                //System.out.println("No value for key "+key);
                out.writeBoolean(true);
            } else  {
                out.writeBoolean(false);
                //System.out.println("Value type: "+value.getClass().getSimpleName()+"; Type="+type.getValue()+"; Key="+key);
                writeValue(out, type.getValue(), value);
            }
        }
    }

    private void writeValue(DataOutput out, Integer type, Object value) throws IOException {
        switch(type) {
            case BOOL: out.writeBoolean((Boolean)value); break;
            case BYTE: out.writeByte((Byte)value); break;
            case CHAR: out.writeChar((Character)value); break;
            case DOUBLE:
                if(((Double)value).isNaN())
                    out.writeBoolean(true); //Consider as NULL
                else
                    out.writeDouble((Double)value); break;
            case FLOAT: out.writeFloat((Float)value); break;
            case INT: out.writeInt((Integer)value); break;
            case LONG: out.writeLong((Long)value); break;
            case SHORT: out.writeShort((Short)value); break;
            case STRING: Text.writeString(out, (String)value); break;
            case TEXT: Text.writeString(out, escapeJava((String) value)); break;
        }
    }

    public void write(PreparedStatement out) throws SQLException {
        int i=1;
        for(Map.Entry<String,Integer> type : getTypes().entrySet()) {
            String key = type.getKey();
            Object value = get(key);

            //Write stuff
            writeValue(out, i++, type.getValue(), value);
        }
    }

    private void writeValue(PreparedStatement out, int i, Integer type, Object value) throws SQLException {
        switch(type) {
            case BOOL: out.setBoolean(i, (Boolean) value); break;
            case BYTE: out.setByte(i, (Byte) value); break;
            case CHAR: out.setString(i, value.toString()); break;
            case DOUBLE: out.setDouble(i, (Double)value); break;
            case FLOAT: out.setFloat(i, (Float)value); break;
            case INT: out.setInt(i, (Integer)value); break;
            case LONG: out.setLong(i, (Long)value); break;
            case SHORT: out.setShort(i, (Short)value); break;
            case STRING: out.setString(i, (String)value); break;
            case TEXT: out.setString(i, (String) value); break;
        }
    }

    public abstract Map<String, Integer> getTypes();

    public void readFields(DataInput in) throws IOException {
        for(Map.Entry<String,Integer> type : getTypes().entrySet()) {
            if(in.readBoolean())   //NULL
                put(type.getKey(), null);
            else    //NOT NULL
                put(type.getKey(), readValue(in, type.getValue()));
        }
    }

    private Object readValue(DataInput in, Integer type) throws IOException {
        switch(type) {
            case BOOL: return in.readBoolean();
            case BYTE: return in.readByte();
            case CHAR: return in.readChar();
            case DOUBLE: return in.readDouble();
            case FLOAT: return in.readFloat();
            case INT: return in.readInt();
            case LONG: return in.readLong();
            case SHORT: return in.readShort();
            case STRING: return Text.readString(in);
            case TEXT: return unescapeJava(Text.readString(in));
        }
        return null;
    }

    public void readFields(ResultSet in) throws SQLException {
        int i = 1;
        for(Map.Entry<String,Integer> type : getTypes().entrySet())
            put(type.getKey(), readValue(in, i, type.getValue()));
    }

    private Object readValue(ResultSet in, int i, Integer type) throws SQLException {
        switch(type) {
            case BOOL: return in.getBoolean(i);
            case BYTE: return in.getByte(i);
            case CHAR: return in.getString(i);
            case DOUBLE: return in.getDouble(i);
            case FLOAT: return in.getFloat(i);
            case INT: return in.getInt(i);
            case LONG: return in.getLong(i);
            case SHORT: return in.getShort(i);
            case STRING: return in.getString(i);
            case TEXT: return in.getString(i);
        }
        return null;
    }

    public Integer getInteger(String key) {
        return (Integer)get(key);
    }

    public Boolean getBool(String key) {
        return (Boolean)get(key);
    }

    public Boolean is(String key) {
        return getBool(key);
    }

    public String getString(String key) {
        return (String)get(key);
    }

    public Long getLong(String key) {
        return (Long)get(key);
    }

    public Double getDouble(String key) {
        return (Double)get(key);
    }
    
    public Byte getByte(String key) {
        return (Byte)get(key);
    }

    @Override
    public Object put(String key, Object value) {
        if(safeMode) {
            Class type;
            if(!getTypes().containsKey(key))
                throw new ClassCastException("Unknown field "+key);
            switch(getTypes().get(key)) {
                case BOOL: return super.put(key, (Boolean)value);
                case BYTE: return super.put(key, (Byte)value);
                case CHAR: return super.put(key, (Character)value);
                case DOUBLE: return super.put(key, (Double)value);
                case FLOAT: return super.put(key, (Float)value);
                case INT: return super.put(key, (Integer)value);
                case LONG:
                    if(value instanceof BigInteger)
                        value = ((BigInteger)value).longValue();
                    return super.put(key, (Long)value);
                case SHORT: return super.put(key, (Short)value);
                case STRING: return super.put(key, (String)value);
                case TEXT: return super.put(key, (String)value);
                default: throw new ClassCastException("Unknown type for key "+key);
            }
        } else
            return super.put(key, value);
    }

    public static <R extends HadoopRecord> R buildRecord(R record, ResultSet result) throws SQLException {
        ResultSetMetaData meta = result.getMetaData();
        for(int i=1; i<=meta.getColumnCount(); i++) {
            String key = meta.getColumnName(i);
            if(record.getTypes().containsKey(key))
                record.put(key, result.getObject(key));
        }
        return record;
    }
}
